using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using UnityEngine;

namespace TextToMotion.Editor
{
    /// <summary>
    /// Wraps Unity Sentis to run one ONNX text-to-motion model.
    ///
    /// ADAPT BEFORE USE:
    ///   1. Set INPUT_* / OUTPUT_* to match your model's tensor names (check in Netron).
    ///   2. Replace TokenisePrompt() with your real tokeniser (CLIP / BPE / etc.).
    ///   3. Adjust the scheduler in RunAsync() if your model expects DDIM/PLMS.
    ///
    /// Sidecar JSON (same name as .onnx, extension .json):
    ///   { "max_frames": 196, "fps": 20.0, "mean_file": "Mean.npy", "std_file": "Std.npy" }
    /// </summary>
    public sealed class MotionInferenceRunner : IDisposable
    {
        // ── Edit these to match your model (open it in https://netron.app) ──
        private const string INPUT_TOKENS    = "text_tokens";    // int32  [1, seq]
        private const string INPUT_NOISE     = "noisy_motion";   // float  [1, F, 263]
        private const string INPUT_TIMESTEP  = "timestep";       // int32  [1]
        private const string OUTPUT_MOTION   = "motion_pred";    // float  [1, F, 263]

        private const int   DEFAULT_FRAMES   = 196;
        private const float DEFAULT_FPS      = 20f;
        private const int   MAX_TOKEN_LEN    = 32;

        public int   MaxFrames { get; private set; } = DEFAULT_FRAMES;
        public float Fps       { get; private set; } = DEFAULT_FPS;

        private Unity.InferenceEngine.Model   _model;
        private IWorker _worker;
        private string  _meanPath, _stdPath;
        private float[] _mean, _std;
        private bool    _normReady;

        public MotionInferenceRunner(string onnxPath)
        {
            _model  = Unity.InferenceEngine.ModelLoader.Load(onnxPath);
            _worker = WorkerFactory.CreateWorker(Unity.InferenceEngine.BackendType.GPUCompute, _model);

            string json = Path.ChangeExtension(onnxPath, ".json");
            if (File.Exists(json)) LoadSidecar(json);
        }

        public async Task<float[]> RunAsync(
            string            prompt,
            int               steps,
            int               seed,
            IProgress<float>  progress,
            CancellationToken ct)
        {
            int[] tokens = TokenisePrompt(prompt);
            float[] noisy = GaussianNoise(MaxFrames * 263, seed);

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

        private float[] StepModel(int[] tokens, float[] noisy, int timestep)
        {
            using var tTok = new TensorInt(new Unity.InferenceEngine.TensorShape(1, tokens.Length), tokens);
            using var tNoise = new TensorFloat(new Unity.InferenceEngine.TensorShape(1, MaxFrames, 263), noisy);
            using var tStep = new TensorInt(new Unity.InferenceEngine.TensorShape(1), new[] { timestep });

            _worker.SetInput(INPUT_TOKENS,   tTok);
            _worker.SetInput(INPUT_NOISE,    tNoise);
            _worker.SetInput(INPUT_TIMESTEP, tStep);
            _worker.Schedule();

            var output = _worker.PeekOutput(OUTPUT_MOTION) as TensorFloat;
            output?.MakeReadable();
            return output?.ToReadOnlyArray() ?? noisy;
        }

        // ── Tokeniser stub — replace with your real tokeniser ──────────────
        private static int[] TokenisePrompt(string prompt)
        {
            var ids = new int[MAX_TOKEN_LEN];
            int n = Math.Min(prompt.Length, MAX_TOKEN_LEN);
            for (int i = 0; i < n; i++) ids[i] = prompt[i];
            return ids;
        }

        // ── Denormalisation ────────────────────────────────────────────────
        private float[] Denormalise(float[] raw)
        {
            if (!_normReady) LoadNorm();
            if (_mean == null || _std == null) return raw;

            int frames = raw.Length / 263;
            var result = new float[raw.Length];
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

        private static float[] LoadNpy(string path)
        {
            try
            {
                byte[] b = File.ReadAllBytes(path);
                int hLen = b[8] | (b[9] << 8);
                int off  = 10 + hLen;
                var arr  = new float[(b.Length - off) / 4];
                Buffer.BlockCopy(b, off, arr, 0, arr.Length * 4);
                return arr;
            }
            catch (Exception e) { Debug.LogWarning($"[TTM] .npy load failed: {e.Message}"); return null; }
        }

        // ── Sidecar JSON ───────────────────────────────────────────────────
        [Serializable] private class Sidecar
        {
            public int   max_frames = DEFAULT_FRAMES;
            public float fps        = DEFAULT_FPS;
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

        // ── Gaussian noise ────────────────────────────────────────────────
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

        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;
        }
    }
}