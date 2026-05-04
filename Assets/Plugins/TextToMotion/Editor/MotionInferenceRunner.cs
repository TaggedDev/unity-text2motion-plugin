// Editor/MotionInferenceRunner.cs
// Unity AI Inference Engine 2.6.1
// Shapes verified:
//   xt          : (1, 60, 263)  float32
//   t           : (1,)          int64   ← важно!
//   text_emb    : (1, 512)      float32
//   mask        : (1, 263)      float32
//   pred_x0     : (1, 60, 263)  float32
//   clip input  : (1, 77)       int32
//   clip output : (1, 512)      float32

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;
using Tokenizers.DotNet;

namespace TextToMotion.Editor
{
    public sealed class MotionInferenceRunner : IDisposable
    {
        // ── Verified tensor names ─────────────────────────────────────────────
        private const string D_IN_XT       = "xt";
        private const string D_IN_T        = "t";
        private const string D_IN_TEXT_EMB = "text_emb";
        private const string D_IN_MASK     = "mask";
        private const string D_OUT_PRED    = "pred_x0";
        private const string C_IN_TOKENS   = "input";
        private const string C_OUT_EMBED   = "output";

        // ── Verified constants ────────────────────────────────────────────────
        private const int   VERIFIED_FRAMES = 60;
        private const int   EMBED_DIM       = 512;
        private const int   MOTION_DIM      = 263;
        private const int   TOKEN_LEN       = 77;
        private const int   SOT_TOKEN       = 49406;
        private const int   EOT_TOKEN       = 49407;
        private const float DEFAULT_FPS     = 20f;

        public int   MaxFrames { get; private set; } = VERIFIED_FRAMES;
        public float Fps       { get; private set; } = DEFAULT_FPS;

        // ── Sentis workers ────────────────────────────────────────────────────
        private Model  _denoiserModel;
        private Worker _denoiserWorker;
        private Model  _clipModel;
        private Worker _clipWorker;

        private Tokenizer _tokenizer;   // null → char fallback

        // ── Normalisation ─────────────────────────────────────────────────────
        private float[] _mean;
        private float[] _std;
        private bool    _normLoaded;
        private string  _meanPath, _stdPath;

        // ── Constructor (MUST run on main thread) ─────────────────────────────

        public MotionInferenceRunner(
            ModelAsset denoiserAsset,
            ModelAsset clipAsset,
            string     sidecarJsonPath   = null,
            string     tokenizerJsonPath = null)
        {
            if (denoiserAsset == null) throw new ArgumentNullException(nameof(denoiserAsset));
            if (clipAsset     == null) throw new ArgumentNullException(nameof(clipAsset));

            _denoiserModel  = ModelLoader.Load(denoiserAsset);
            _denoiserWorker = new Worker(_denoiserModel, BackendType.CPU);

            _clipModel  = ModelLoader.Load(clipAsset);
            _clipWorker = new Worker(_clipModel, BackendType.CPU);

            // Tokenizer
            if (!string.IsNullOrEmpty(tokenizerJsonPath) && File.Exists(tokenizerJsonPath))
            {
                try
                {
                    // Microsoft.ML.Tokenizers: загружаем BPE tokenizer из HuggingFace json
                    _tokenizer = new Tokenizer(vocabPath: tokenizerJsonPath);
                    Debug.Log($"[TTM] Tokenizer loaded: {tokenizerJsonPath}");
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[TTM] Tokenizer load failed, using fallback: {e.Message}");
                    _tokenizer = null;
                }
            }
            else
            {
                Debug.LogWarning("[TTM] tokenizer.json not found — char-level fallback active.");
            }

            if (!string.IsNullOrEmpty(sidecarJsonPath) && File.Exists(sidecarJsonPath))
                LoadSidecar(sidecarJsonPath);
        }

        // ── RunAsync — ONLY ON MAIN THREAD ────────────────────────────────────
        // Task.Yield() между шагами = editor не фризит, но Sentis остаётся на main thread

        public async Task<float[]> RunAsync(
            string           prompt,
            int              steps,
            int              seed,
            IProgress<float> progress,
            CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            progress?.Report(0.02f);

            // CLIP encode — sync, main thread
            float[] textEmb = EncodeText(prompt);
            progress?.Report(0.10f);

            // Init noise
            float[] xt   = GaussianNoise(MaxFrames * MOTION_DIM, seed);
            float[] mask = BuildMask();

            // Diffusion denoising loop
            for (int i = 0; i < steps; i++)
            {
                ct.ThrowIfCancellationRequested();

                xt = StepDenoiser(xt, i, steps, textEmb, mask);

                progress?.Report(0.10f + (float)(i + 1) / steps * 0.80f);

                // Yield to editor every 5 steps so UI stays responsive
                if (i % 5 == 0)
                    await Task.Yield();
            }

            progress?.Report(0.92f);
            float[] result = Denormalise(xt);
            progress?.Report(1.00f);
            return result;
        }

        // ── CLIP encode ───────────────────────────────────────────────────────

        private float[] EncodeText(string prompt)
        {
            int[] tokens = BuildTokens(prompt);

            // CLIP input dtype = int32 (dtype_code=6)
            using var tTokens = new Tensor<int>(new TensorShape(1, TOKEN_LEN), tokens);
            _clipWorker.SetInput(C_IN_TOKENS, tTokens);
            _clipWorker.Schedule();

            var output = _clipWorker.PeekOutput(C_OUT_EMBED) as Tensor<float>;
            if (output == null)
            {
                Debug.LogError("[TTM] CLIP output tensor is null. Check C_OUT_EMBED name.");
                return new float[EMBED_DIM];
            }

            float[] result = output.DownloadToArray();

            // Verified shape: (1, 512) → length = 512
            if (result.Length != EMBED_DIM)
            {
                Debug.LogWarning($"[TTM] CLIP output length={result.Length}, expected {EMBED_DIM}. Truncating/padding.");
                var fixed_ = new float[EMBED_DIM];
                Array.Copy(result, fixed_, Math.Min(result.Length, EMBED_DIM));
                return fixed_;
            }

            return result;
        }

        // ── Token builder ─────────────────────────────────────────────────────

        private int[] BuildTokens(string prompt)
        {
            var ids = new int[TOKEN_LEN];

            if (_tokenizer != null)
            {
                try
                {
                    // sappho192/Tokenizers.DotNet API: Encode возвращает uint[]
                    uint[] encoded = _tokenizer.Encode(prompt);

                    ids[0] = SOT_TOKEN;
                    int n = Math.Min(encoded.Length, TOKEN_LEN - 2);
                    for (int i = 0; i < n; i++)
                        ids[i + 1] = (int)encoded[i];
                    ids[n + 1] = EOT_TOKEN;
                    // остаток = 0 (padding)

                    return ids;
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[TTM] Tokenizer.Encode failed, char fallback: {e.Message}");
                }
            }

            // Char-level fallback
            ids[0] = SOT_TOKEN;
            int fn = Math.Min(prompt.Length, TOKEN_LEN - 2);
            for (int i = 0; i < fn; i++)
                ids[i + 1] = Math.Min((int)prompt[i], 49405);
            ids[fn + 1] = EOT_TOKEN;
            return ids;
        }

        // ── Denoiser step ─────────────────────────────────────────────────────

        private float[] StepDenoiser(
            float[] xt,
            int     stepIdx,
            int     totalSteps,
            float[] textEmb,
            float[] mask)
        {
            // t: linear schedule 999 → 0
            long t = (long)(999f * (totalSteps - 1 - stepIdx) / Math.Max(totalSteps - 1, 1));

            // xt: (1, frames, 263)  float32
            using var tXt = new Tensor<float>(
                new TensorShape(1, MaxFrames, MOTION_DIM), xt);

            // t: (1,) int64  ← модель ждёт int64 (dtype_code=7)
            using var tT = new Tensor<int>(new TensorShape(1));
            tT[0] = (int)t;

            // text_emb: (1, 512)  float32
            using var tEmb = new Tensor<float>(
                new TensorShape(1, EMBED_DIM), textEmb);

            // mask: (1, 263)  float32
            using var tMsk = new Tensor<float>(
                new TensorShape(1, MOTION_DIM), mask);

            _denoiserWorker.SetInput(D_IN_XT,       tXt);
            _denoiserWorker.SetInput(D_IN_T,        tT);
            _denoiserWorker.SetInput(D_IN_TEXT_EMB, tEmb);
            _denoiserWorker.SetInput(D_IN_MASK,     tMsk);
            _denoiserWorker.Schedule();

            var output = _denoiserWorker.PeekOutput(D_OUT_PRED) as Tensor<float>;
            if (output == null)
            {
                Debug.LogWarning($"[TTM] Denoiser output null at step {stepIdx}");
                return xt;
            }

            return output.DownloadToArray();
        }

        // ── Mask ──────────────────────────────────────────────────────────────

        private static float[] BuildMask()
        {
            var m = new float[MOTION_DIM];
            for (int i = 0; i < m.Length; i++) m[i] = 1f;
            return m;
        }

        // ── Denormalise: x_norm * std + mean ─────────────────────────────────

        private float[] Denormalise(float[] raw)
        {
            if (!_normLoaded) LoadNorm();

            // Если нет mean/std — возвращаем как есть
            if (_mean == null || _std == null)
            {
                Debug.LogWarning("[TTM] mean/std not loaded — skipping denormalisation.");
                return raw;
            }

            int frames = raw.Length / MOTION_DIM;
            var result = new float[raw.Length];

            for (int f = 0; f < frames; f++)
            {
                int off = f * MOTION_DIM;
                for (int d = 0; d < MOTION_DIM; d++)
                    result[off + d] = raw[off + d] * _std[d] + _mean[d];
            }
            return result;
        }

        private void LoadNorm()
        {
            _normLoaded = true;
            if (_meanPath != null && File.Exists(_meanPath))
                _mean = LoadNpy(_meanPath);
            else
                Debug.LogWarning($"[TTM] mean.npy not found: {_meanPath}");

            if (_stdPath != null && File.Exists(_stdPath))
                _std = LoadNpy(_stdPath);
            else
                Debug.LogWarning($"[TTM] std.npy not found: {_stdPath}");
        }

        // ── .npy loader (float32 only, C-contiguous) ──────────────────────────

        private static float[] LoadNpy(string path)
        {
            try
            {
                byte[] b = File.ReadAllBytes(path);

                // Magic: 0x93 'N' 'U' 'M' 'P' 'Y'
                if (b.Length < 10 || b[0] != 0x93)
                    throw new InvalidDataException("Not a valid .npy file");

                // Header length at bytes 8-9 (little-endian)
                int hLen  = b[8] | (b[9] << 8);
                int dataOffset = 10 + hLen;
                int count = (b.Length - dataOffset) / sizeof(float);

                var arr = new float[count];
                Buffer.BlockCopy(b, dataOffset, arr, 0, count * sizeof(float));
                Debug.Log($"[TTM] Loaded .npy: {Path.GetFileName(path)}  count={count}");
                return arr;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[TTM] .npy load failed ({path}): {e.Message}");
                return null;
            }
        }

        // ── Sidecar JSON ──────────────────────────────────────────────────────

        [Serializable]
        private class Sidecar
        {
            public int    max_frames = VERIFIED_FRAMES;
            public float  fps        = DEFAULT_FPS;
            public string mean_file;
            public string std_file;
        }

        private void LoadSidecar(string jsonPath)
        {
            try
            {
                var d = JsonUtility.FromJson<Sidecar>(File.ReadAllText(jsonPath));
                MaxFrames  = d.max_frames;
                Fps        = d.fps;
                string dir = Path.GetDirectoryName(jsonPath);
                if (!string.IsNullOrEmpty(d.mean_file))
                    _meanPath = Path.Combine(dir, d.mean_file);
                if (!string.IsNullOrEmpty(d.std_file))
                    _stdPath  = Path.Combine(dir, d.std_file);
                Debug.Log($"[TTM] Sidecar loaded: frames={MaxFrames} fps={Fps}");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[TTM] Sidecar error: {e.Message}");
            }
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
                arr[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1))
                               * Math.Sin(2.0 * Math.PI * u2));
            }
            return arr;
        }

        // ── Dispose ───────────────────────────────────────────────────────────

        public void Dispose()
        {
            _denoiserWorker?.Dispose();
            _clipWorker?.Dispose();
            _denoiserModel = null;
            _clipModel     = null;
        }
    }
}