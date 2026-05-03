using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
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
            w.minSize = new Vector2(400, 520);
        }

        private const string K_FOLDER = "TTM_Folder";
        private const string K_MODEL  = "TTM_Model";
        private const string K_OUTPUT = "TTM_Output";
        private const string K_STEPS  = "TTM_Steps";
        private const string K_SEED   = "TTM_Seed";

        private List<string>          _onnxPaths = new();
        private string                _selectedOnnx;
        private MotionInferenceRunner _runner;
        private CancellationTokenSource _cts;
        private bool                  _running;

        private SkeletonPreview  _preview;
        private IMGUIContainer   _previewContainer;

        private TextField     _folderField;
        private DropdownField _modelDrop;
        private TextField     _promptField;
        private SliderInt     _stepsSlider;
        private IntegerField  _seedField;
        private TextField     _outputField;
        private Button        _generateBtn;
        private Button        _cancelBtn;
        private ProgressBar   _progressBar;
        private Label         _statusLabel;

        public void CreateGUI()
        {
            _preview = new SkeletonPreview();
            BuildUI();
        }

        private void BuildUI()
        {
            var root = rootVisualElement;
            root.style.paddingTop = root.style.paddingLeft =
            root.style.paddingRight = root.style.paddingBottom = 10;

            // --- Models folder ---
            root.Add(Label("Models Folder"));
            var folderRow = Row();
            _folderField = new TextField { isReadOnly = true };
            _folderField.style.flexGrow = 1;
            var browseModels = new Button(() => BrowseFolder()) { text = "Browse…" };
            browseModels.style.width = 70;
            folderRow.Add(_folderField);
            folderRow.Add(browseModels);
            root.Add(folderRow);

            // --- Model dropdown ---
            root.Add(Label("Model"));
            _modelDrop = new DropdownField();
            root.Add(_modelDrop);

            // --- Prompt ---
            root.Add(Label("Prompt"));
            _promptField = new TextField { multiline = true };
            _promptField.style.height = 68;
            root.Add(_promptField);

            // --- Steps + Seed ---
            var settingsRow = Row();
            settingsRow.style.marginTop = 4;
            var leftCol = new VisualElement(); leftCol.style.flexGrow = 1;
            var rightCol = new VisualElement(); rightCol.style.flexGrow = 1; rightCol.style.marginLeft = 8;
            _stepsSlider = new SliderInt("Steps", 10, 200) { value = EditorPrefs.GetInt(K_STEPS, 50), showInputField = true };
            _seedField = new IntegerField("Seed") { value = EditorPrefs.GetInt(K_SEED, 42) };
            leftCol.Add(_stepsSlider);
            rightCol.Add(_seedField);
            settingsRow.Add(leftCol);
            settingsRow.Add(rightCol);
            root.Add(settingsRow);

            // --- Output folder ---
            root.Add(Label("Output Folder"));
            var outRow = Row();
            _outputField = new TextField { value = EditorPrefs.GetString(K_OUTPUT, "Assets/GeneratedMotions") };
            _outputField.style.flexGrow = 1;
            var browseOut = new Button(() => BrowseOutput()) { text = "Browse…" };
            browseOut.style.width = 70;
            outRow.Add(_outputField);
            outRow.Add(browseOut);
            root.Add(outRow);

            // --- Buttons ---
            var btnRow = Row();
            btnRow.style.marginTop = 8;
            _generateBtn = new Button(() => _ = GenerateAsync()) { text = "▶  Generate" };
            _generateBtn.style.flexGrow = 1;
            _generateBtn.style.height = 30;
            _cancelBtn = new Button(Cancel) { text = "✕" };
            _cancelBtn.style.width = 34;
            _cancelBtn.style.marginLeft = 6;
            btnRow.Add(_generateBtn);
            btnRow.Add(_cancelBtn);
            root.Add(btnRow);

            // --- Progress ---
            _progressBar = new ProgressBar { lowValue = 0, highValue = 100 };
            _progressBar.style.marginTop = 6;
            root.Add(_progressBar);

            _statusLabel = new Label("Select a models folder to begin.");
            _statusLabel.style.fontSize = 11;
            _statusLabel.style.marginTop = 3;
            _statusLabel.style.color = new StyleColor(new Color(0.6f, 0.6f, 0.6f));
            root.Add(_statusLabel);

            // --- Preview ---
            root.Add(Label("Preview"));
            _previewContainer = new IMGUIContainer();
            _previewContainer.style.height = 200;
            _previewContainer.style.backgroundColor = new StyleColor(new Color(0.13f, 0.13f, 0.13f));
            _previewContainer.style.marginTop = 4;
            _previewContainer.onGUIHandler = () =>
            {
                if (_preview.Draw(_previewContainer.contentRect))
                    Repaint();
            };
            root.Add(_previewContainer);

            // --- Restore state ---
            _folderField.value = EditorPrefs.GetString(K_FOLDER, "");
            if (!string.IsNullOrEmpty(_folderField.value))
                ScanFolder(_folderField.value);

            _modelDrop.RegisterValueChangedCallback(e => SelectModel(e.newValue));
            _stepsSlider.RegisterValueChangedCallback(e => EditorPrefs.SetInt(K_STEPS, e.newValue));
            _seedField.RegisterValueChangedCallback(e => EditorPrefs.SetInt(K_SEED, e.newValue));

            RefreshButtons();
        }

        private static VisualElement Row()
        {
            var e = new VisualElement();
            e.style.flexDirection = FlexDirection.Row;
            e.style.alignItems = Align.Center;
            e.style.marginBottom = 4;
            return e;
        }

        private static Label Label(string text)
        {
            var l = new Label(text);
            l.style.fontSize = 11;
            l.style.color = new StyleColor(new Color(0.67f, 0.67f, 0.67f));
            l.style.marginTop = 6;
            l.style.marginBottom = 2;
            return l;
        }

        // ── Folder / model ───────────────────────────────────────────────

        private void BrowseFolder()
        {
            string path = EditorUtility.OpenFolderPanel("Models folder", "", "");
            if (string.IsNullOrEmpty(path)) return;
            _folderField.value = path;
            EditorPrefs.SetString(K_FOLDER, path);
            ScanFolder(path);
        }

        private void BrowseOutput()
        {
            string path = EditorUtility.OpenFolderPanel("Output folder", Application.dataPath, "");
            if (string.IsNullOrEmpty(path)) return;
            if (path.StartsWith(Application.dataPath))
                path = "Assets" + path.Substring(Application.dataPath.Length);
            _outputField.value = path;
            EditorPrefs.SetString(K_OUTPUT, path);
        }

        private void ScanFolder(string folder)
        {
            _onnxPaths.Clear();
            if (!Directory.Exists(folder)) { SetStatus("Folder not found."); return; }

            _onnxPaths = Directory.GetFiles(folder, "*.onnx", SearchOption.TopDirectoryOnly)
                                  .OrderBy(x => x).ToList();

            if (_onnxPaths.Count == 0)
            {
                _modelDrop.choices = new List<string> { "— no models —" };
                _modelDrop.value = _modelDrop.choices[0];
                _modelDrop.SetEnabled(false);
                SetStatus("No .onnx files found.");
                return;
            }

            var names = _onnxPaths.Select(Path.GetFileNameWithoutExtension).ToList();
            _modelDrop.choices = names;
            _modelDrop.SetEnabled(true);

            string last = EditorPrefs.GetString(K_MODEL, "");
            _modelDrop.value = names.Contains(last) ? last : names[0];
            SelectModel(_modelDrop.value);
            SetStatus($"{_onnxPaths.Count} model(s) loaded.");
        }

        private void SelectModel(string displayName)
        {
            _runner?.Dispose();
            _runner = null;
            _selectedOnnx = _onnxPaths.FirstOrDefault(
                p => Path.GetFileNameWithoutExtension(p) == displayName);
            if (!string.IsNullOrEmpty(displayName))
                EditorPrefs.SetString(K_MODEL, displayName);
            RefreshButtons();
        }

        // ── Generate ─────────────────────────────────────────────────────

        private async Task GenerateAsync()
        {
            string prompt = _promptField.value?.Trim() ?? "";
            if (string.IsNullOrEmpty(prompt)) { SetStatus("⚠ Enter a prompt."); return; }
            if (string.IsNullOrEmpty(_selectedOnnx)) { SetStatus("⚠ No model selected."); return; }

            _running = true;
            _cts = new CancellationTokenSource();
            RefreshButtons();
            SetProgress(0f);
            SetStatus("Loading model…");

            try
            {
                if (_runner == null)
                    _runner = new MotionInferenceRunner(_selectedOnnx);

                var progress = new Progress<float>(p =>
                {
                    SetProgress(p);
                    SetStatus(p < 0.85f ? $"Denoising… {(int)(p * 100)}%" : "Decoding…");
                });

                float[] raw = await _runner.RunAsync(
                    prompt, _stepsSlider.value, _seedField.value, progress, _cts.Token);

                SetStatus("Saving clip…");

                string dir   = _outputField.value;
                string name  = SafeName(prompt);
                string asset = $"{dir}/{name}_{DateTime.Now:yyyyMMdd_HHmmss}.anim";

                var clip = HumanML3DToClip.Build(raw, _runner.MaxFrames, _runner.Fps, asset);

                _preview.SetData(raw, _runner.MaxFrames, _runner.Fps);
                Repaint();

                SetProgress(1f);
                SetStatus($"✓ {asset}");
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

        // ── Helpers ──────────────────────────────────────────────────────

        private void SetStatus(string msg)  => _statusLabel.text = msg;
        private void SetProgress(float v)   => _progressBar.value = v * 100f;
        private void RefreshButtons()
        {
            _generateBtn.SetEnabled(!_running && !string.IsNullOrEmpty(_selectedOnnx));
            _cancelBtn.SetEnabled(_running);
        }

        private static string SafeName(string s)
        {
            var bad = Path.GetInvalidFileNameChars();
            string r = new string(s.Select(c => bad.Contains(c) ? '_' : c).ToArray()).Replace(' ', '_');
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