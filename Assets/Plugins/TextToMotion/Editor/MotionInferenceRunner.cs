// Editor/MotionInferenceRunner.cs
// Requires: com.unity.ai.inference 2.6.1

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace TextToMotion.Editor
{
    public sealed class MotionInferenceRunner : IDisposable
    {
        // ── Denoiser tensor names (из Netron) ────────────────────────────────
        private const string D_IN_XT       = "xt";
        private const string D_IN_T        = "t";
        private const string D_IN_TEXT_EMB = "text_emb";
        private const string D_IN_MASK     = "mask";
        private const string D_OUT_PRED    = "pred_x0";

        // ── CLIP tensor names (из Netron) ────────────────────────────────────
        private const string C_IN_TOKENS = "input";
        private const string C_OUT_EMBED = "output";

        private const int   DEFAULT_FRAMES = 196;
        private const float DEFAULT_FPS    = 20f;

        public int   MaxFrames { get; private set; } = DEFAULT_FRAMES;
        public float Fps       { get; private set; } = DEFAULT_FPS;

        private Model  _denoiserModel;
        private Worker _denoiserWorker;
        private Model  _clipModel;
        private Worker _clipWorker;

        private ClipTokenizer _tokenizer;

        private string  _meanPath, _stdPath;
        private float[] _mean, _std;
        private bool    _normReady;

        // ── Constructor ──────────────────────────────────────────────────────

        public MotionInferenceRunner(
            ModelAsset denoiserAsset,
            ModelAsset clipAsset,
            string     sidecarJsonPath   = null,
            string     tokenizerJsonPath = null)
        {
            if (denoiserAsset == null) throw new ArgumentNullException(nameof(denoiserAsset));
            if (clipAsset     == null) throw new ArgumentNullException(nameof(clipAsset));

            _denoiserModel  = ModelLoader.Load(denoiserAsset);
            _denoiserWorker = new Worker(_denoiserModel, BackendType.GPUCompute);

            _clipModel  = ModelLoader.Load(clipAsset);
            _clipWorker = new Worker(_clipModel, BackendType.GPUCompute);

            if (!string.IsNullOrEmpty(tokenizerJsonPath) && File.Exists(tokenizerJsonPath))
            {
                try   { _tokenizer = ClipTokenizer.Load(tokenizerJsonPath); }
                catch (Exception e) { Debug.LogWarning($"[TTM] Tokenizer load failed: {e.Message}"); }
            }

            if (_tokenizer == null)
                Debug.LogWarning("[TTM] tokenizer.json not found — using char-level fallback.");

            if (!string.IsNullOrEmpty(sidecarJsonPath) && File.Exists(sidecarJsonPath))
                LoadSidecar(sidecarJsonPath);
        }

        // ── Main entry point ─────────────────────────────────────────────────

        public async Task<float[]> RunAsync(
            string            prompt,
            int               steps,
            int               seed,
            IProgress<float>  progress,
            CancellationToken ct)
        {
            progress?.Report(0.02f);
            float[] textEmb = await Task.Run(() => EncodeText(prompt), ct);

            float[] xt   = GaussianNoise(MaxFrames * 263, seed);
            float[] mask = BuildMask();

            for (int i = 0; i < steps; i++)
            {
                ct.ThrowIfCancellationRequested();
                progress?.Report(0.05f + i / (float)steps * 0.80f);

                int t = (int)(999f * (steps - 1 - i) / Math.Max(steps - 1, 1));
                xt = await Task.Run(() => StepDenoiser(xt, t, textEmb, mask), ct);
            }

            progress?.Report(0.90f);
            float[] result = Denormalise(xt);
            progress?.Report(1.00f);
            return result;
        }

        // ── CLIP encode ──────────────────────────────────────────────────────

        private float[] EncodeText(string prompt)
        {
            int[] tokens = _tokenizer != null
                ? _tokenizer.Encode(prompt)
                : FallbackTokenise(prompt);

            using var tTokens = new Tensor<int>(new TensorShape(1, 77), tokens);
            _clipWorker.SetInput(C_IN_TOKENS, tTokens);
            _clipWorker.Schedule();

            var output = _clipWorker.PeekOutput(C_OUT_EMBED) as Tensor<float>;
            if (output == null)
            {
                Debug.LogError("[TTM] CLIP output is null. Check C_OUT_EMBED name in Netron.");
                return new float[512];
            }

            return output.DownloadToArray();
        }

        // ── Denoiser step ────────────────────────────────────────────────────

        private float[] StepDenoiser(float[] xt, int t, float[] textEmb, float[] mask)
        {
            using var tXt  = new Tensor<float>(new TensorShape(1, MaxFrames, 263), xt);
            using var tT   = new Tensor<int>  (new TensorShape(1), new[] { t });
            using var tEmb = new Tensor<float>(new TensorShape(1, 512), textEmb);
            using var tMsk = new Tensor<float>(new TensorShape(1, 263), mask);

            _denoiserWorker.SetInput(D_IN_XT,       tXt);
            _denoiserWorker.SetInput(D_IN_T,        tT);
            _denoiserWorker.SetInput(D_IN_TEXT_EMB, tEmb);
            _denoiserWorker.SetInput(D_IN_MASK,     tMsk);
            _denoiserWorker.Schedule();

            var output = _denoiserWorker.PeekOutput(D_OUT_PRED) as Tensor<float>;
            if (output == null) return xt;

            return output.DownloadToArray();
        }

        // ── Tokeniser fallback ───────────────────────────────────────────────

        private static int[] FallbackTokenise(string prompt)
        {
            const int SOT = 49406, EOT = 49407;
            var ids = new int[77];
            ids[0] = SOT;
            int n = Math.Min(prompt.Length, 75);
            for (int i = 0; i < n; i++)
                ids[i + 1] = (int)Math.Min((int)prompt[i], 49405);  // ← фикс Clamp ambiguity
            ids[n + 1] = EOT;
            return ids;
        }

        // ── Mask ─────────────────────────────────────────────────────────────

        private static float[] BuildMask()
        {
            var m = new float[263];
            for (int i = 0; i < m.Length; i++) m[i] = 1f;
            return m;
        }

        // ── Denormalise ──────────────────────────────────────────────────────

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

        private static float[] LoadNpy(string path)
        {
            try
            {
                byte[] b = File.ReadAllBytes(path);
                if (b.Length < 10 || b[0] != 0x93)
                    throw new InvalidDataException("Not a valid .npy file");
                int hLen  = b[8] | (b[9] << 8);
                int off   = 10 + hLen;
                int count = (b.Length - off) / 4;
                var arr   = new float[count];
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
                var    d   = JsonUtility.FromJson<Sidecar>(File.ReadAllText(jsonPath));
                MaxFrames  = d.max_frames;
                Fps        = d.fps;
                string dir = Path.GetDirectoryName(jsonPath);
                if (!string.IsNullOrEmpty(d.mean_file)) _meanPath = Path.Combine(dir, d.mean_file);
                if (!string.IsNullOrEmpty(d.std_file))  _stdPath  = Path.Combine(dir, d.std_file);
            }
            catch (Exception e) { Debug.LogWarning($"[TTM] Sidecar error: {e.Message}"); }
        }

        // ── Gaussian noise ────────────────────────────────────────────────────

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

        // ── Dispose ──────────────────────────────────────────────────────────

        public void Dispose()
        {
            _denoiserWorker?.Dispose();
            _clipWorker?.Dispose();
            _denoiserModel = null;
            _clipModel     = null;
        }
    }
}