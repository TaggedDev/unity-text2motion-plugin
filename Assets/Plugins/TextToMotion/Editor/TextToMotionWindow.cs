// Editor/TextToMotionWindow.cs
// Unity AI Inference Engine 2.6.1 — модели лежат в Assets/Text2MotionModels/*.onnx

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
            w.minSize = new Vector2(420, 560);
        }

        // ── EditorPrefs keys ─────────────────────────────────────────────────
        private const string K_MODELS_FOLDER = "TTM_ModelsFolder";
        private const string K_MODEL         = "TTM_Model";
        private const string K_OUTPUT        = "TTM_Output";
        private const string K_STEPS         = "TTM_Steps";
        private const string K_SEED          = "TTM_Seed";

        // Папка по умолчанию внутри проекта
        private const string DEFAULT_MODELS_FOLDER = "Assets/Text2MotionModels";

        // ── State ────────────────────────────────────────────────────────────
        // asset paths: "Assets/Text2MotionModels/model.onnx"
        private readonly List<string> _assetPaths = new();
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

            // ── Models folder (внутри Assets/) ──────────────────────────────
            root.Add(MakeLabel("Models Folder (inside Assets/)"));
            var folderRow = MakeRow();
            _modelsFolderField = new TextField { isReadOnly = true };
            _modelsFolderField.style.flexGrow = 1;
            var refreshBtn = new Button(RefreshModelList) { text = "↺ Refresh" };
            refreshBtn.style.width = 70;
            folderRow.Add(_modelsFolderField);
            folderRow.Add(refreshBtn);
            root.Add(folderRow);

            // info label
            var hint = new Label("Place .onnx files in Assets/Text2MotionModels/ and click Refresh");
            hint.style.fontSize = 10;
            hint.style.color    = new StyleColor(new Color(0.5f, 0.5f, 0.5f));
            hint.style.marginBottom = 4;
            root.Add(hint);

            // ── Model dropdown ───────────────────────────────────────────────
            root.Add(MakeLabel("Model"));
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

            _statusLabel = new Label("Click ↺ Refresh to scan Assets/Text2MotionModels/");
            _statusLabel.style.fontSize    = 11;
            _statusLabel.style.marginTop   = 3;
            _statusLabel.style.color       = new StyleColor(new Color(0.6f, 0.6f, 0.6f));
            _statusLabel.style.whiteSpace  = WhiteSpace.Normal;
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

            // ── Wire up callbacks ────────────────────────────────────────────
            _modelDrop.RegisterValueChangedCallback(e => SelectModelByName(e.newValue));
            _stepsSlider.RegisterValueChangedCallback(e => EditorPrefs.SetInt(K_STEPS, e.newValue));
            _seedField.RegisterValueChangedCallback(e => EditorPrefs.SetInt(K_SEED, e.newValue));
            _outputField.RegisterValueChangedCallback(e => EditorPrefs.SetString(K_OUTPUT, e.newValue));

            // ── Restore last state ───────────────────────────────────────────
            string savedFolder = EditorPrefs.GetString(K_MODELS_FOLDER, DEFAULT_MODELS_FOLDER);
            _modelsFolderField.value = savedFolder;
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
            l.style.fontSize    = 11;
            l.style.color       = new StyleColor(new Color(0.67f, 0.67f, 0.67f));
            l.style.marginTop   = 6;
            l.style.marginBottom = 2;
            return l;
        }

        // ── Model list ───────────────────────────────────────────────────────

        /// <summary>
        /// Сканирует папку через AssetDatabase (только Assets/).
        /// Вызывается кнопкой Refresh и при старте окна.
        /// </summary>
        private void RefreshModelList()
        {
            string folder = _modelsFolderField.value;
            if (string.IsNullOrEmpty(folder))
                folder = DEFAULT_MODELS_FOLDER;

            EditorPrefs.SetString(K_MODELS_FOLDER, folder);

            // Убеждаемся, что папка существует
            if (!AssetDatabase.IsValidFolder(folder))
            {
                // Пытаемся создать, чтобы пользователь сразу мог туда положить файлы
                string parent = Path.GetDirectoryName(folder)?.Replace('\\', '/') ?? "Assets";
                string leaf   = Path.GetFileName(folder);
                if (AssetDatabase.IsValidFolder(parent))
                    AssetDatabase.CreateFolder(parent, leaf);

                SetStatus($"Folder '{folder}' created. Place .onnx files there.");
                _assetPaths.Clear();
                UpdateDropdown(null);
                return;
            }

            // Ищем все ModelAsset (.onnx → импортируется Unity как ModelAsset)
            string[] guids = AssetDatabase.FindAssets("t:ModelAsset", new[] { folder });

            _assetPaths.Clear();
            foreach (string guid in guids)
                _assetPaths.Add(AssetDatabase.GUIDToAssetPath(guid));

            _assetPaths.Sort();

            if (_assetPaths.Count == 0)
            {
                SetStatus($"No .onnx models found in {folder}");
                UpdateDropdown(null);
                return;
            }

            string lastModel = EditorPrefs.GetString(K_MODEL, "");
            UpdateDropdown(lastModel);
            SetStatus($"{_assetPaths.Count} model(s) found in {folder}");
        }

        private void UpdateDropdown(string preferredName)
        {
            if (_assetPaths.Count == 0)
            {
                _modelDrop.choices = new List<string> { "— no models —" };
                _modelDrop.value   = _modelDrop.choices[0];
                _modelDrop.SetEnabled(false);
                _selectedAssetPath = null;
                RefreshButtons();
                return;
            }

            var names = _assetPaths
                .Select(p => Path.GetFileNameWithoutExtension(p))
                .ToList();

            _modelDrop.choices = names;
            _modelDrop.SetEnabled(true);

            // Восстанавливаем последнюю выбранную модель
            string select = names.Contains(preferredName) ? preferredName : names[0];
            _modelDrop.value = select;
            SelectModelByName(select);
        }

        private void SelectModelByName(string displayName)
        {
            // Сбрасываем runner при смене модели
            _runner?.Dispose();
            _runner = null;

            _selectedAssetPath = _assetPaths.FirstOrDefault(
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

            // Конвертируем абсолютный путь в Assets-relative
            if (path.StartsWith(Application.dataPath))
                path = "Assets" + path.Substring(Application.dataPath.Length);

            _outputField.value = path;
            EditorPrefs.SetString(K_OUTPUT, path);
        }

        // ── Generate ─────────────────────────────────────────────────────────

        private async Task GenerateAsync()
        {
            string prompt = _promptField.value?.Trim() ?? "";
            if (string.IsNullOrEmpty(prompt))
            {
                SetStatus("⚠ Enter a prompt.");
                return;
            }
            if (string.IsNullOrEmpty(_selectedAssetPath))
            {
                SetStatus("⚠ No model selected. Click Refresh.");
                return;
            }

            _running = true;
            _cts     = new CancellationTokenSource();
            RefreshButtons();
            SetProgress(0f);
            SetStatus("Loading model…");

            try
            {
                // Lazy-init runner — пересоздаём только при смене модели
                if (_runner == null)
                {
                    var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(_selectedAssetPath);
                    if (modelAsset == null)
                    {
                        SetStatus($"✗ Cannot load ModelAsset: {_selectedAssetPath}");
                        return;
                    }

                    // Ищем sidecar JSON рядом с .onnx на диске
                    string fullPath = Path.GetFullPath(_selectedAssetPath);
                    string jsonPath = Path.ChangeExtension(fullPath, ".json");

                    _runner = new MotionInferenceRunner(modelAsset, jsonPath);
                }

                var progress = new Progress<float>(p =>
                {
                    SetProgress(p);
                    SetStatus(p < 0.85f
                        ? $"Denoising… {(int)(p * 100)}%"
                        : "Decoding motion…");
                });

                float[] raw = await _runner.RunAsync(
                    prompt,
                    _stepsSlider.value,
                    _seedField.value,
                    progress,
                    _cts.Token);

                SetStatus("Building AnimationClip…");

                // Убеждаемся, что output папка существует
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
            catch (OperationCanceledException)
            {
                SetStatus("Cancelled.");
                SetProgress(0f);
            }
            catch (Exception ex)
            {
                SetStatus($"✗ {ex.Message}");
                Debug.LogException(ex);
                SetProgress(0f);
            }
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

        private void SetProgress(float v) => _progressBar.value = v * 100f;

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