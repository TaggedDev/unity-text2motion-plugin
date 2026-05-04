// Editor/TextToMotionWindow.cs
// Unity AI Inference Engine 2.6.1

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace TextToMotion.Editor
{
    public class TextToMotionWindow : EditorWindow
    {
        [MenuItem("Tools/Text To Motion")]
        public static void Open()
        {
            var w = GetWindow<TextToMotionWindow>();
            w.titleContent = new GUIContent("Text To Motion");
            w.minSize = new Vector2(420, 580);
        }

        // ── EditorPrefs keys ─────────────────────────────────────────────────
        private const string K_MODELS_FOLDER = "TTM_ModelsFolder";
        private const string K_MODEL         = "TTM_Model";
        private const string K_OUTPUT        = "TTM_Output";
        private const string K_STEPS         = "TTM_Steps";
        private const string K_SEED          = "TTM_Seed";

        private const string DEFAULT_MODELS_FOLDER = "Assets/Text2MotionModels";

        // ── State ────────────────────────────────────────────────────────────
        // Только denoiser-пути (без clip)
        private readonly List<string> _denoiserPaths = new();
        private string                _selectedAssetPath;
        private MotionInferenceRunner _runner;
        private CancellationTokenSource _cts;
        private bool                  _running;

        // ── UI refs ──────────────────────────────────────────────────────────
        private SkeletonPreview  _preview;
        private IMGUIContainer   _previewContainer;

        private TextField     _modelsFolderField;
        private DropdownField _modelDrop;
        private TextField     _promptField;
        private SliderInt     _stepsSlider;
        private IntegerField  _seedField;
        private TextField     _outputField;
        private Button        _generateBtn;
        private Button        _cancelBtn;
        private ProgressBar   _progressBar;
        private Label         _statusLabel;

        // ────────────────────────────────────────────────────────────────────
        public void CreateGUI()
        {
            _preview = new SkeletonPreview();
            BuildUI();
        }

        private void BuildUI()
        {
            var root = rootVisualElement;
            root.style.paddingTop    = 10;
            root.style.paddingLeft   = 10;
            root.style.paddingRight  = 10;
            root.style.paddingBottom = 10;

            // ── Models folder ────────────────────────────────────────────────
            root.Add(MakeLabel("Models Folder (inside Assets/)"));
            var folderRow = MakeRow();
            _modelsFolderField = new TextField { isReadOnly = true };
            _modelsFolderField.style.flexGrow = 1;
            var refreshBtn = new Button(RefreshModelList) { text = "↺ Refresh" };
            refreshBtn.style.width = 70;
            folderRow.Add(_modelsFolderField);
            folderRow.Add(refreshBtn);
            root.Add(folderRow);

            var hint = new Label("Put denoiser + clip*.onnx + tokenizer.json in the folder, then Refresh");
            hint.style.fontSize     = 10;
            hint.style.color        = new StyleColor(new Color(0.5f, 0.5f, 0.5f));
            hint.style.whiteSpace   = WhiteSpace.Normal;
            hint.style.marginBottom = 4;
            root.Add(hint);

            // ── Model dropdown ───────────────────────────────────────────────
            root.Add(MakeLabel("Denoiser Model"));
            _modelDrop = new DropdownField();
            root.Add(_modelDrop);

            // ── Prompt ───────────────────────────────────────────────────────
            root.Add(MakeLabel("Prompt"));
            _promptField = new TextField { multiline = true };
            _promptField.style.height = 68;
            root.Add(_promptField);

            // ── Steps + Seed ─────────────────────────────────────────────────
            var settingsRow = MakeRow();
            settingsRow.style.marginTop = 4;
            var leftCol  = new VisualElement(); leftCol.style.flexGrow  = 1;
            var rightCol = new VisualElement(); rightCol.style.flexGrow = 1;
            rightCol.style.marginLeft = 8;

            _stepsSlider = new SliderInt("Steps", 10, 200)
            {
                value          = EditorPrefs.GetInt(K_STEPS, 50),
                showInputField = true
            };
            _seedField = new IntegerField("Seed") { value = EditorPrefs.GetInt(K_SEED, 42) };

            leftCol.Add(_stepsSlider);
            rightCol.Add(_seedField);
            settingsRow.Add(leftCol);
            settingsRow.Add(rightCol);
            root.Add(settingsRow);

            // ── Output folder ────────────────────────────────────────────────
            root.Add(MakeLabel("Output Folder"));
            var outRow = MakeRow();
            _outputField = new TextField
            {
                value = EditorPrefs.GetString(K_OUTPUT, "Assets/GeneratedMotions")
            };
            _outputField.style.flexGrow = 1;
            var browseOut = new Button(BrowseOutput) { text = "Browse…" };
            browseOut.style.width = 70;
            outRow.Add(_outputField);
            outRow.Add(browseOut);
            root.Add(outRow);

            // ── Generate / Cancel ────────────────────────────────────────────
            var btnRow = MakeRow();
            btnRow.style.marginTop = 8;
            _generateBtn = new Button(() => _ = GenerateAsync()) { text = "▶  Generate" };
            _generateBtn.style.flexGrow = 1;
            _generateBtn.style.height   = 30;
            _cancelBtn = new Button(Cancel) { text = "✕" };
            _cancelBtn.style.width      = 34;
            _cancelBtn.style.marginLeft = 6;
            btnRow.Add(_generateBtn);
            btnRow.Add(_cancelBtn);
            root.Add(btnRow);

            // ── Progress ─────────────────────────────────────────────────────
            _progressBar = new ProgressBar { lowValue = 0, highValue = 100 };
            _progressBar.style.marginTop = 6;
            root.Add(_progressBar);

            _statusLabel = new Label("Click ↺ Refresh to scan models folder.");
            _statusLabel.style.fontSize   = 11;
            _statusLabel.style.marginTop  = 3;
            _statusLabel.style.color      = new StyleColor(new Color(0.6f, 0.6f, 0.6f));
            _statusLabel.style.whiteSpace = WhiteSpace.Normal;
            root.Add(_statusLabel);

            // ── Preview ──────────────────────────────────────────────────────
            root.Add(MakeLabel("Preview"));
            _previewContainer = new IMGUIContainer();
            _previewContainer.style.height          = 200;
            _previewContainer.style.backgroundColor = new StyleColor(new Color(0.13f, 0.13f, 0.13f));
            _previewContainer.style.marginTop       = 4;
            _previewContainer.onGUIHandler = () =>
            {
                if (_preview.Draw(_previewContainer.contentRect))
                    Repaint();
            };
            root.Add(_previewContainer);

            // ── Callbacks ────────────────────────────────────────────────────
            _modelDrop.RegisterValueChangedCallback(e => SelectModelByName(e.newValue));
            _stepsSlider.RegisterValueChangedCallback(e => EditorPrefs.SetInt(K_STEPS, e.newValue));
            _seedField.RegisterValueChangedCallback(e => EditorPrefs.SetInt(K_SEED, e.newValue));
            _outputField.RegisterValueChangedCallback(e => EditorPrefs.SetString(K_OUTPUT, e.newValue));

            // ── Restore ──────────────────────────────────────────────────────
            _modelsFolderField.value = EditorPrefs.GetString(K_MODELS_FOLDER, DEFAULT_MODELS_FOLDER);
            RefreshModelList();
            RefreshButtons();
        }

        // ── UI helpers ───────────────────────────────────────────────────────

        private static VisualElement MakeRow()
        {
            var e = new VisualElement();
            e.style.flexDirection = FlexDirection.Row;
            e.style.alignItems    = Align.Center;
            e.style.marginBottom  = 4;
            return e;
        }

        private static Label MakeLabel(string text)
        {
            var l = new Label(text);
            l.style.fontSize     = 11;
            l.style.color        = new StyleColor(new Color(0.67f, 0.67f, 0.67f));
            l.style.marginTop    = 6;
            l.style.marginBottom = 2;
            return l;
        }

        // ── Model list ───────────────────────────────────────────────────────

        private void RefreshModelList()
        {
            string folder = _modelsFolderField.value;
            if (string.IsNullOrEmpty(folder))
                folder = DEFAULT_MODELS_FOLDER;

            EditorPrefs.SetString(K_MODELS_FOLDER, folder);

            // Создаём папку если нет
            if (!AssetDatabase.IsValidFolder(folder))
            {
                string parent = Path.GetDirectoryName(folder)?.Replace('\\', '/') ?? "Assets";
                string leaf   = Path.GetFileName(folder);
                if (AssetDatabase.IsValidFolder(parent))
                    AssetDatabase.CreateFolder(parent, leaf);

                SetStatus($"Folder '{folder}' created. Place .onnx files there and click Refresh.");
                _denoiserPaths.Clear();
                UpdateDropdown(null);
                return;
            }

            // Все ModelAsset в папке
            string[] guids = AssetDatabase.FindAssets("t:ModelAsset", new[] { folder });
            var allPaths = guids
                .Select(AssetDatabase.GUIDToAssetPath)
                .OrderBy(p => p)
                .ToList();

            // Clip отдельно — деноизеры отдельно
            var clipPaths = allPaths
                .Where(p => Path.GetFileNameWithoutExtension(p)
                    .ToLowerInvariant().Contains("clip"))
                .ToList();

            _denoiserPaths.Clear();
            foreach (var p in allPaths)
            {
                if (!clipPaths.Contains(p))
                    _denoiserPaths.Add(p);
            }

            // Статус
            string clipStatus = clipPaths.Count > 0
                ? $"CLIP: {Path.GetFileNameWithoutExtension(clipPaths[0])}"
                : "⚠ No clip*.onnx found";

            if (_denoiserPaths.Count == 0)
            {
                SetStatus($"No denoiser models found. {clipStatus}");
                UpdateDropdown(null);
                return;
            }

            UpdateDropdown(EditorPrefs.GetString(K_MODEL, ""));
            SetStatus($"{_denoiserPaths.Count} denoiser(s) | {clipStatus}");
        }

        private void UpdateDropdown(string preferredName)
        {
            if (_denoiserPaths.Count == 0)
            {
                _modelDrop.choices = new List<string> { "— no models —" };
                _modelDrop.value   = _modelDrop.choices[0];
                _modelDrop.SetEnabled(false);
                _selectedAssetPath = null;
                RefreshButtons();
                return;
            }

            var names = _denoiserPaths
                .Select(p => Path.GetFileNameWithoutExtension(p))
                .ToList();

            _modelDrop.choices = names;
            _modelDrop.SetEnabled(true);

            string select = names.Contains(preferredName) ? preferredName : names[0];
            _modelDrop.value = select;
            SelectModelByName(select);
        }

        private void SelectModelByName(string displayName)
        {
            _runner?.Dispose();
            _runner = null;

            _selectedAssetPath = _denoiserPaths.FirstOrDefault(
                p => Path.GetFileNameWithoutExtension(p) == displayName);

            if (!string.IsNullOrEmpty(displayName))
                EditorPrefs.SetString(K_MODEL, displayName);

            RefreshButtons();
        }

        // ── Output browse ────────────────────────────────────────────────────

        private void BrowseOutput()
        {
            string path = EditorUtility.OpenFolderPanel("Output folder", Application.dataPath, "");
            if (string.IsNullOrEmpty(path)) return;

            if (path.StartsWith(Application.dataPath))
                path = "Assets" + path.Substring(Application.dataPath.Length);

            _outputField.value = path;
            EditorPrefs.SetString(K_OUTPUT, path);
        }

        // ── Generate ─────────────────────────────────────────────────────────

        private async Task GenerateAsync()
        {
            string prompt = _promptField.value?.Trim() ?? "";
            if (string.IsNullOrEmpty(prompt))      { SetStatus("⚠ Enter a prompt."); return; }
            if (string.IsNullOrEmpty(_selectedAssetPath)) { SetStatus("⚠ No model selected."); return; }

            _running = true;
            _cts     = new CancellationTokenSource();
            RefreshButtons();
            SetProgress(0f);
            SetStatus("Loading models…");

            try
            {
                if (_runner == null)
                {
                    string folder = _modelsFolderField.value;

                    // Denoiser
                    var denoiserAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(_selectedAssetPath);
                    if (denoiserAsset == null)
                    {
                        SetStatus($"✗ Cannot load denoiser: {_selectedAssetPath}");
                        return;
                    }

                    // CLIP — ищем в той же папке
                    string clipPath = AssetDatabase.FindAssets("t:ModelAsset", new[] { folder })
                        .Select(AssetDatabase.GUIDToAssetPath)
                        .FirstOrDefault(p => Path.GetFileNameWithoutExtension(p)
                            .ToLowerInvariant().Contains("clip"));

                    if (string.IsNullOrEmpty(clipPath))
                    {
                        SetStatus($"✗ CLIP model not found in {folder}. Add clip*.onnx.");
                        return;
                    }

                    var clipAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(clipPath);
                    if (clipAsset == null)
                    {
                        SetStatus($"✗ Cannot load CLIP asset: {clipPath}");
                        return;
                    }

                    // Пути к sidecar и tokenizer на диске
                    string fullDenoiserPath = Path.GetFullPath(_selectedAssetPath);
                    string sidecarPath      = Path.ChangeExtension(fullDenoiserPath, ".json");
                    string tokenizerPath    = Path.Combine(
                        Path.GetDirectoryName(fullDenoiserPath) ?? "", "tokenizer.json");

                    _runner = new MotionInferenceRunner(
                        denoiserAsset, clipAsset, sidecarPath, tokenizerPath);
                }

                var progress = new Progress<float>(p =>
                {
                    SetProgress(p);
                    SetStatus(p < 0.85f
                        ? $"Denoising… {(int)(p * 100)}%"
                        : "Decoding motion…");
                });

                float[] raw = await _runner.RunAsync(
                    prompt, _stepsSlider.value, _seedField.value, progress, _cts.Token);

                SetStatus("Building AnimationClip…");

                // Создаём output папку если нет
                string outDir = _outputField.value;
                if (!AssetDatabase.IsValidFolder(outDir))
                {
                    string parent = Path.GetDirectoryName(outDir)?.Replace('\\', '/') ?? "Assets";
                    string leaf   = Path.GetFileName(outDir);
                    AssetDatabase.CreateFolder(parent, leaf);
                }

                string safeName  = SafeName(prompt);
                string assetPath = $"{outDir}/{safeName}_{DateTime.Now:yyyyMMdd_HHmmss}.anim";

                var clip = HumanML3DToClip.Build(raw, _runner.MaxFrames, _runner.Fps, assetPath);

                _preview.SetData(raw, _runner.MaxFrames, _runner.Fps);
                Repaint();

                SetProgress(1f);
                SetStatus($"✓ Saved: {assetPath}");
                EditorGUIUtility.PingObject(clip);
            }
            catch (OperationCanceledException) { SetStatus("Cancelled."); SetProgress(0f); }
            catch (Exception ex)               { SetStatus($"✗ {ex.Message}"); Debug.LogException(ex); SetProgress(0f); }
            finally
            {
                _running = false;
                _cts?.Dispose();
                _cts = null;
                RefreshButtons();
            }
        }

        private void Cancel() => _cts?.Cancel();

        // ── Helpers ──────────────────────────────────────────────────────────

        private void SetStatus(string msg) => _statusLabel.text = msg;
        private void SetProgress(float v)  => _progressBar.value = v * 100f;

        private void RefreshButtons()
        {
            bool canGenerate = !_running && !string.IsNullOrEmpty(_selectedAssetPath);
            _generateBtn.SetEnabled(canGenerate);
            _cancelBtn.SetEnabled(_running);
        }

        private static string SafeName(string s)
        {
            char[] bad = Path.GetInvalidFileNameChars();
            string r   = new string(s.Select(c => bad.Contains(c) ? '_' : c).ToArray())
                             .Replace(' ', '_');
            return r.Length > 40 ? r[..40] : r;
        }

        private void OnDisable()
        {
            _cts?.Cancel();
            _runner?.Dispose();
            _preview?.Dispose();
        }
    }
}