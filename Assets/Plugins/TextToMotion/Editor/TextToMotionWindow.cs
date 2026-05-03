using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace TextToMotion.Editor
{
    public class TextToMotionWindow : EditorWindow
    {
        private const string UxmlPath = "Assets/Plugins/TextToMotion/Editor/TextToMotionWindow.uxml";
        private const string UssPath = "Assets/Plugins/TextToMotion/Editor/TextToMotionWindow.uss";

        [MenuItem("Tools/Text To Motion")]
        public static void ShowWindow()
        {
            var window = GetWindow<TextToMotionWindow>();
            window.titleContent = new GUIContent("Text To Motion");
            window.minSize = new Vector2(460, 420);
        }

        public void CreateGUI()
        {
            rootVisualElement.Clear();

            var uxml = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(UxmlPath);
            var uss = AssetDatabase.LoadAssetAtPath<StyleSheet>(UssPath);

            if (uxml == null)
            {
                rootVisualElement.Add(new Label($"UXML not found at: {UxmlPath}"));
                return;
            }

            if (uss != null)
            {
                rootVisualElement.styleSheets.Add(uss);
            }

            var tree = uxml.Instantiate();
            rootVisualElement.Add(tree);

            BindPlaceholderBehaviour();
        }

        private void BindPlaceholderBehaviour()
        {
            var modelsFolderField = rootVisualElement.Q<TextField>("models-folder-field");
            var modelDropdown = rootVisualElement.Q<DropdownField>("model-dropdown");
            var outputFolderField = rootVisualElement.Q<TextField>("output-folder-field");
            var browseModelsButton = rootVisualElement.Q<Button>("browse-models-folder-button");
            var browseOutputButton = rootVisualElement.Q<Button>("browse-output-folder-button");
            var generateButton = rootVisualElement.Q<Button>("generate-button");
            var statusLabel = rootVisualElement.Q<Label>("status-label");

            if (modelsFolderField != null)
                modelsFolderField.value = "Not selected";

            if (outputFolderField != null)
                outputFolderField.value = "Assets";

            if (modelDropdown != null)
            {
                modelDropdown.choices = new System.Collections.Generic.List<string> { "No models yet" };
                modelDropdown.value = "No models yet";
                modelDropdown.SetEnabled(false);
            }

            if (browseModelsButton != null)
            {
                browseModelsButton.clicked += () =>
                {
                    statusLabel.text = "Models folder selection will be implemented next.";
                };
            }

            if (browseOutputButton != null)
            {
                browseOutputButton.clicked += () =>
                {
                    statusLabel.text = "Output folder selection will be implemented next.";
                };
            }

            if (generateButton != null)
            {
                generateButton.SetEnabled(false);
                generateButton.clicked += () =>
                {
                    statusLabel.text = "Generate pipeline is not connected yet.";
                };
            }
        }
    }
}