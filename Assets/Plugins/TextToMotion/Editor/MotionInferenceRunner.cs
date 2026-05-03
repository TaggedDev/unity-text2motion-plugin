// Editor/MotionInferenceRunner.cs
// Requires: com.unity.ai.inference 2.2.0+  (namespace Unity.InferenceEngine)
// Tested against 2.6.1 — 2026-04-02

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;          // ← было Unity.Sentis
using UnityEngine;

namespace TextToMotion.Editor
{
    /// <summary>
    /// Запускает одну ONNX text-to-motion модель через Unity Sentis / AI Inference Engine 2.6+.
    ///
    /// ПЕРЕД ИСПОЛЬЗОВАНИЕМ:
    ///   1. Проверь имена тензоров своей модели в https://netron.app и поправь константы INPUT_*/OUTPUT_*.
    ///   2. Замени TokenisePrompt() на настоящий токенайзер (CLIP/BPE).
    ///      В 2.5+ доступен Unity.InferenceEngine.Tokenization — рассмотри его.
    ///   3. Если модель использует DDIM/PLMS, скорректируй scheduler в RunAsync().
    ///
    /// Sidecar JSON (то же имя, расширение .json рядом с .onnx):
    ///   { "max_frames": 196, "fps": 20.0, "mean_file": "Mean.npy", "std_file": "Std.npy" }
    /// </summary>
    public sealed class MotionInferenceRunner : IDisposable
    {
        // ── Имена тензоров — открой модель в Netron и сверь ─────────────────
        private const string INPUT_TOKENS   = "text_tokens";   // int32  [1, seq]
        private const string INPUT_NOISE    = "noisy_motion";  // float32 [1, F, 263]
        private const string INPUT_TIMESTEP = "timestep";      // int32  [1]
        private const string OUTPUT_MOTION  = "motion_pred";   // float32 [1, F, 263]

        private const int   DEFAULT_FRAMES  = 196;
        private const float DEFAULT_FPS     = 20f;
        private const int   MAX_TOKEN_LEN   = 32;

        public int   MaxFrames { get; private set; } = DEFAULT_FRAMES;
        public float Fps       { get; private set; } = DEFAULT_FPS;

        // ── 2.6.1: Model + Worker (нет IWorker / WorkerFactory) ─────────────
        private Model  _model;
        private Worker _worker;   // конкретный класс, не интерфейс

        private string  _meanPath, _stdPath;
        private float[] _mean, _std;
        private bool    _normReady;

        public MotionInferenceRunner(string onnxPath)
        {
            _model  = ModelLoader.Load(onnxPath);
            // Worker(Model, BackendType) — актуальный конструктор в 2.x
            _worker = new Worker(_model, BackendType.GPUCompute);

            string json = Path.ChangeExtension(onnxPath, ".json");
            if (File.Exists(json)) LoadSidecar(json);
        }

        // ────────────────────────────────────────────────────────────────────
        //  Основной entry point
        // ────────────────────────────────────────────────────────────────────
        public async Task<float[]> RunAsync(
            string           prompt,
            int              steps,
            int              seed,
            IProgress<float> progress,
            CancellationToken ct)
        {
            int[]   tokens = TokenisePrompt(prompt);
            float[] noisy  = GaussianNoise(MaxFrames * 263, seed);

            for (int i = 0; i < steps; i++)
            {
                ct.ThrowIfCancellationRequested();
                progress?.Report(i / (float)steps * 0.85f);

                int t = (int)(999f * (steps - i) / steps);   // linear scheduler
                noisy = await Task.Run(() => StepModel(tokens, noisy, t), ct);
            }

            progress?.Report(0.9f);
            float[] result = Denormalise(noisy);
            progress?.Report(1f);
            return result;
        }

        // ────────────────────────────────────────────────────────────────────
        //  Один шаг диффузии
        //  В 2.6.1:
        //    • TensorInt   → Tensor<int>
        //    • TensorFloat → Tensor<float>
        //    • new TensorFloat(shape, data[])  → new Tensor<float>(shape, data[])
        //    • worker.PeekOutput(name)         → worker.PeekOutput<float>(name)  (generic)
        //    • output.MakeReadable()           → DownloadToArray() / ReadbackAndClone()
        // ────────────────────────────────────────────────────────────────────
        private float[] StepModel(int[] tokens, float[] noisy, int timestep)
        {
            // Создаём тензоры через generic Tensor<T>
            using var tTok   = new Tensor<int>  (new TensorShape(1, tokens.Length), tokens);
            using var tNoise = new Tensor<float>(new TensorShape(1, MaxFrames, 263), noisy);
            using var tStep  = new Tensor<int>  (new TensorShape(1), new[] { timestep });

            _worker.SetInput(INPUT_TOKENS,   tTok);
            _worker.SetInput(INPUT_NOISE,    tNoise);
            _worker.SetInput(INPUT_TIMESTEP, tStep);
            _worker.Schedule();

            // PeekOutput возвращает Tensor (non-generic base); каст к Tensor<float>
            var raw = _worker.PeekOutput(OUTPUT_MOTION) as Tensor<float>;
            if (raw == null) return noisy;

            // В 2.x для чтения данных с GPU нужен явный readback:
            // DownloadToArray() блокирует до завершения GPU-работы и возвращает float[]
            float[] data = raw.DownloadToArray();
            return data;
        }

        // ── Токенайзер-заглушка — замени на настоящий ───────────────────────
        private static int[] TokenisePrompt(string prompt)
        {
            var ids = new int[MAX_TOKEN_LEN];
            int n   = Math.Min(prompt.Length, MAX_TOKEN_LEN);
            for (int i = 0; i < n; i++) ids[i] = prompt[i];
            return ids;
        }

        // ── Денормализация ───────────────────────────────────────────────────
        private float[] Denormalise(float[] raw)
        {
            if (!_normReady) LoadNorm();
            if (_mean == null || _std == null) return raw;

            int    frames = raw.Length / 263;
            var    result = new float[raw.Length];
            for (int f = 0; f < frames; f++)
            {
                int off = f * 263;
                for (int d = 0; d < 263; d++)
                    result[off + d] = raw[off + d] * _std[d] + _mean[d];
            }
            return result;
        }

        private void LoadNorm()
        {
            _normReady = true;
            if (_meanPath != null && File.Exists(_meanPath)) _mean = LoadNpy(_meanPath);
            if (_stdPath  != null && File.Exists(_stdPath))  _std  = LoadNpy(_stdPath);
        }

        // ── Минимальный парсер .npy (float32, C-order, без fortran-order) ───
        private static float[] LoadNpy(string path)
        {
            try
            {
                byte[] b = File.ReadAllBytes(path);
                // .npy magic: \x93NUMPY (6 bytes) + major/minor (2) + header_len (2, little-endian)
                if (b.Length < 10 || b[0] != 0x93)
                    throw new InvalidDataException("Not a valid .npy file");

                int hLen = b[8] | (b[9] << 8);
                int off  = 10 + hLen;
                int count = (b.Length - off) / 4;
                var arr  = new float[count];
                Buffer.BlockCopy(b, off, arr, 0, count * 4);
                return arr;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[TTM] .npy load failed ({path}): {e.Message}");
                return null;
            }
        }

        // ── Sidecar JSON ─────────────────────────────────────────────────────
        [Serializable]
        private class Sidecar
        {
            public int    max_frames = DEFAULT_FRAMES;
            public float  fps        = DEFAULT_FPS;
            public string mean_file;
            public string std_file;
        }

        private void LoadSidecar(string jsonPath)
        {
            try
            {
                var d   = JsonUtility.FromJson<Sidecar>(File.ReadAllText(jsonPath));
                MaxFrames = d.max_frames;
                Fps       = d.fps;
                string dir = Path.GetDirectoryName(jsonPath);
                if (!string.IsNullOrEmpty(d.mean_file)) _meanPath = Path.Combine(dir, d.mean_file);
                if (!string.IsNullOrEmpty(d.std_file))  _stdPath  = Path.Combine(dir, d.std_file);
            }
            catch (Exception e) { Debug.LogWarning($"[TTM] Sidecar error: {e.Message}"); }
        }

        // ── Gaussian noise (Box-Muller) ──────────────────────────────────────
        private static float[] GaussianNoise(int n, int seed)
        {
            var rng = new System.Random(seed);
            var arr = new float[n];
            for (int i = 0; i < n; i++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                arr[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            }
            return arr;
        }

        // ── IDisposable ──────────────────────────────────────────────────────
        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;
            // Model не имеет Dispose() в публичном API — просто обнуляем
            _model = null;
        }
    }
}