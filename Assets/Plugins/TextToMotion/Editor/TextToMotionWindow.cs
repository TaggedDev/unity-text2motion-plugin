using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace TextToMotion.Editor
{
    public class TextToMotionWindow : EditorWindow
    {
        private const string ModelsFolderPrefKey = "TextToMotion.ModelsFolder";
        private const string OutputFolderPrefKey = "TextToMotion.OutputFolder";
        private const string PromptPrefKey = "TextToMotion.Prompt";
        private const string SelectedModelPrefKey = "TextToMotion.SelectedModel";

        private string _modelsFolderAbsolutePath;
        private string _outputFolderProjectRelativePath = "Assets";
        private string _selectedModelFileName;
        private string _promptText;

        private List<string> _modelFileNames = new();

        private TextField _modelsFolderField;
        private DropdownField _modelDropdown;
        private TextField _promptField;
        private TextField _outputFolderField;
        private Button _generateButton;
        private Label _statusLabel;

        [MenuItem("Tools/Text To Motion")]
        public static void ShowWindow()
        {
            var window = GetWindow<TextToMotionWindow>();
            window.titleContent = new GUIContent("Text To Motion");
            window.minSize = new Vector2(520, 320);
        }

        private void OnEnable()
        {
            _modelsFolderAbsolutePath = EditorPrefs.GetString(ModelsFolderPrefKey, string.Empty);
            _outputFolderProjectRelativePath = EditorPrefs.GetString(OutputFolderPrefKey, "Assets");
            _promptText = EditorPrefs.GetString(PromptPrefKey, string.Empty);
            _selectedModelFileName = EditorPrefs.GetString(SelectedModelPrefKey, string.Empty);

            RefreshModelList();
        }

        public void CreateGUI()
        {
            rootVisualElement.Clear();
            rootVisualElement.style.paddingLeft = 12;
            rootVisualElement.style.paddingRight = 12;
            rootVisualElement.style.paddingTop = 12;
            rootVisualElement.style.paddingBottom = 12;

            var title = new Label("Text to Motion Generator");
            title.style.unityFontStyleAndWeight = FontStyle.Bold;
            title.style.fontSize = 16;
            title.style.marginBottom = 10;
            rootVisualElement.Add(title);

            rootVisualElement.Add(BuildModelsFolderRow());
            rootVisualElement.Add(BuildModelDropdown());
            rootVisualElement.Add(BuildPromptField());
            rootVisualElement.Add(BuildOutputFolderRow());
            rootVisualElement.Add(BuildGenerateButton());
            rootVisualElement.Add(BuildStatusLabel());

            RefreshUIState();
        }

        private VisualElement BuildModelsFolderRow()
        {
            var container = new VisualElement();
            container.style.marginBottom = 10;

            _modelsFolderField = new TextField("Models Folder");
            _modelsFolderField.value = _modelsFolderAbsolutePath;
            _modelsFolderField.isReadOnly = true;
            _modelsFolderField.style.flexGrow = 1;
            _modelsFolderField.AddToClassList("unity-base-field__aligned");

            var browseButton = new Button(OnBrowseModelsFolderClicked)
            {
                text = "Browse..."
            };
            browseButton.style.marginTop = 18;
            browseButton.style.marginLeft = 6;
            browseButton.style.width = 90;

            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.Add(_modelsFolderField);
            row.Add(browseButton);

            container.Add(row);
            return container;
        }

        private VisualElement BuildModelDropdown()
        {
            _modelDropdown = new DropdownField("Model", _modelFileNames, 0);
            _modelDropdown.style.marginBottom = 10;
            _modelDropdown.AddToClassList("unity-base-field__aligned");

            if (_modelFileNames.Count > 0)
            {
                if (!string.IsNullOrEmpty(_selectedModelFileName) && _modelFileNames.Contains(_selectedModelFileName))
                    _modelDropdown.value = _selectedModelFileName;
                else
                    _modelDropdown.value = _modelFileNames[0];
            }
            else
            {
                _modelDropdown.choices = new List<string> { "<No models found>" };
                _modelDropdown.value = "<No models found>";
                _modelDropdown.SetEnabled(false);
            }

            _modelDropdown.RegisterValueChangedCallback(evt =>
            {
                _selectedModelFileName = evt.newValue;
                EditorPrefs.SetString(SelectedModelPrefKey, _selectedModelFileName);
                RefreshUIState();
            });

            return _modelDropdown;
        }

        private VisualElement BuildPromptField()
        {
            _promptField = new TextField("Prompt");
            _promptField.multiline = true;
            _promptField.value = _promptText;
            _promptField.style.height = 110;
            _promptField.style.marginBottom = 10;
            _promptField.AddToClassList("unity-base-field__aligned");

            _promptField.RegisterValueChangedCallback(evt =>
            {
                _promptText = evt.newValue;
                EditorPrefs.SetString(PromptPrefKey, _promptText);
                RefreshUIState();
            });

            return _promptField;
        }

        private VisualElement BuildOutputFolderRow()
        {
            var container = new VisualElement();
            container.style.marginBottom = 12;

            _outputFolderField = new TextField("Output Folder");
            _outputFolderField.value = _outputFolderProjectRelativePath;
            _outputFolderField.isReadOnly = true;
            _outputFolderField.style.flexGrow = 1;
            _outputFolderField.AddToClassList("unity-base-field__aligned");

            var browseButton = new Button(OnBrowseOutputFolderClicked)
            {
                text = "Select..."
            };
            browseButton.style.marginTop = 18;
            browseButton.style.marginLeft = 6;
            browseButton.style.width = 90;

            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.Add(_outputFolderField);
            row.Add(browseButton);

            container.Add(row);
            return container;
        }

        private VisualElement BuildGenerateButton()
        {
            _generateButton = new Button(OnGenerateClicked)
            {
                text = "GENERATE"
            };
            _generateButton.style.height = 32;
            _generateButton.style.marginBottom = 10;
            return _generateButton;
        }

        private VisualElement BuildStatusLabel()
        {
            _statusLabel = new Label("Select folders, choose a model, and enter a prompt.");
            _statusLabel.style.whiteSpace = WhiteSpace.Normal;
            _statusLabel.style.color = new StyleColor(new Color(0.75f, 0.75f, 0.75f));
            return _statusLabel;
        }

        private void OnBrowseModelsFolderClicked()
        {
            var startFolder = string.IsNullOrEmpty(_modelsFolderAbsolutePath)
                ? Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
                : _modelsFolderAbsolutePath;

            var selectedPath = EditorUtility.OpenFolderPanel("Select Models Folder", startFolder, "");

            if (string.IsNullOrWhiteSpace(selectedPath))
                return;

            _modelsFolderAbsolutePath = selectedPath;
            EditorPrefs.SetString(ModelsFolderPrefKey, _modelsFolderAbsolutePath);

            if (_modelsFolderField != null)
                _modelsFolderField.value = _modelsFolderAbsolutePath;

            RefreshModelList();
            RebuildModelDropdown();
            RefreshUIState();
        }

        private void OnBrowseOutputFolderClicked()
        {
            var projectPath = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            var absoluteCurrent = Path.Combine(projectPath, _outputFolderProjectRelativePath);

            var selectedPath = EditorUtility.OpenFolderPanel("Select Output Folder (inside project)", absoluteCurrent, "");

            if (string.IsNullOrWhiteSpace(selectedPath))
                return;

            var normalizedProjectPath = projectPath.Replace("\\", "/");
            var normalizedSelectedPath = Path.GetFullPath(selectedPath).Replace("\\", "/");

            if (!normalizedSelectedPath.StartsWith(normalizedProjectPath, StringComparison.OrdinalIgnoreCase))
            {
                EditorUtility.DisplayDialog(
                    "Invalid Output Folder",
                    "Output folder must be inside the current Unity project.",
                    "OK");
                return;
            }

            var relativePath = normalizedSelectedPath.Substring(normalizedProjectPath.Length).TrimStart('/');

            if (string.IsNullOrEmpty(relativePath))
                relativePath = "Assets";

            _outputFolderProjectRelativePath = relativePath;
            EditorPrefs.SetString(OutputFolderPrefKey, _outputFolderProjectRelativePath);

            if (_outputFolderField != null)
                _outputFolderField.value = _outputFolderProjectRelativePath;

            RefreshUIState();
        }

        private void OnGenerateClicked()
        {
            var selectedModelAbsolutePath = GetSelectedModelAbsolutePath();

            Debug.Log($"[TextToMotion] Generate clicked");
            Debug.Log($"[TextToMotion] Models folder: {_modelsFolderAbsolutePath}");
            Debug.Log($"[TextToMotion] Model: {_selectedModelFileName}");
            Debug.Log($"[TextToMotion] Model path: {selectedModelAbsolutePath}");
            Debug.Log($"[TextToMotion] Prompt: {_promptText}");
            Debug.Log($"[TextToMotion] Output folder: {_outputFolderProjectRelativePath}");

            _statusLabel.text = "Generate clicked. Next step: wire inference pipeline.";
        }

        private void RefreshModelList()
        {
            _modelFileNames.Clear();

            if (string.IsNullOrWhiteSpace(_modelsFolderAbsolutePath))
                return;

            if (!Directory.Exists(_modelsFolderAbsolutePath))
                return;

            _modelFileNames = Directory
                .GetFiles(_modelsFolderAbsolutePath, "*.onnx", SearchOption.TopDirectoryOnly)
                .Select(Path.GetFileName)
                .OrderBy(x => x)
                .ToList();

            if (_modelFileNames.Count == 0)
            {
                _selectedModelFileName = string.Empty;
                EditorPrefs.DeleteKey(SelectedModelPrefKey);
                return;
            }

            if (string.IsNullOrEmpty(_selectedModelFileName) || !_modelFileNames.Contains(_selectedModelFileName))
            {
                _selectedModelFileName = _modelFileNames[0];
                EditorPrefs.SetString(SelectedModelPrefKey, _selectedModelFileName);
            }
        }

        private void RebuildModelDropdown()
        {
            if (_modelDropdown == null)
                return;

            if (_modelDropdown.parent != null)
            {
                var parent = _modelDropdown.parent;
                var index = parent.IndexOf(_modelDropdown);
                parent.RemoveAt(index);

                _modelDropdown = (DropdownField)BuildModelDropdown();
                parent.Insert(index, _modelDropdown);
            }
        }

        private void RefreshUIState()
        {
            bool hasModelsFolder = !string.IsNullOrWhiteSpace(_modelsFolderAbsolutePath) && Directory.Exists(_modelsFolderAbsolutePath);
            bool hasModel = !string.IsNullOrWhiteSpace(_selectedModelFileName);
            bool hasPrompt = !string.IsNullOrWhiteSpace(_promptText);
            bool hasValidOutput = !string.IsNullOrWhiteSpace(_outputFolderProjectRelativePath)
                                  && AssetDatabase.IsValidFolder(_outputFolderProjectRelativePath);

            if (_generateButton != null)
                _generateButton.SetEnabled(hasModelsFolder && hasModel && hasPrompt && hasValidOutput);

            if (_statusLabel == null)
                return;

            if (!hasModelsFolder)
            {
                _statusLabel.text = "Please select a local folder with ONNX models.";
            }
            else if (_modelFileNames.Count == 0)
            {
                _statusLabel.text = "No .onnx files found in selected models folder.";
            }
            else if (!hasPrompt)
            {
                _statusLabel.text = "Enter a text prompt.";
            }
            else if (!hasValidOutput)
            {
                _statusLabel.text = "Select a valid output folder inside the Unity project.";
            }
            else
            {
                _statusLabel.text = "Ready to generate.";
            }
        }

        private string GetSelectedModelAbsolutePath()
        {
            if (string.IsNullOrWhiteSpace(_modelsFolderAbsolutePath) || string.IsNullOrWhiteSpace(_selectedModelFileName))
                return string.Empty;

            return Path.Combine(_modelsFolderAbsolutePath, _selectedModelFileName);
        }
    }
}