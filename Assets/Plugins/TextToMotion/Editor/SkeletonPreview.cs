using UnityEditor;
using UnityEngine;

namespace TextToMotion.Editor
{
    /// <summary>
    /// Draws an animated skeleton wireframe inside an IMGUIContainer
    /// using PreviewRenderUtility + GL lines. No extra scene needed.
    /// </summary>
    public sealed class SkeletonPreview : System.IDisposable
    {
        // HumanML3D 22-joint connections
        private static readonly (int p, int c)[] Bones =
        {
            (0,1),(0,2),(0,3),
            (1,4),(2,5),(3,6),
            (4,7),(5,8),(6,9),
            (7,10),(8,11),(9,12),
            (9,13),(9,14),(12,15),
            (13,16),(14,17),
            (16,18),(17,19),
            (18,20),(19,21)
        };

        private const int FEAT_DIM   = 263;
        private const int POS_OFFSET = 136;   // 4 + 22*6
        private const int NUM_JOINTS = 22;

        private PreviewRenderUtility _pru;
        private float[]  _raw;
        private int      _frames;
        private float    _fps = 20f;
        private double   _startTime;
        private bool     _hasData;

        private static Material _mat;

        public SkeletonPreview()
        {
            _pru = new PreviewRenderUtility();
            _pru.camera.transform.position = new Vector3(0f, 1f, -3.5f);
            _pru.camera.transform.LookAt(Vector3.up);
            _pru.camera.clearFlags      = CameraClearFlags.SolidColor;
            _pru.camera.backgroundColor = new Color(0.13f, 0.13f, 0.13f);
            _pru.camera.nearClipPlane   = 0.1f;
            _pru.camera.farClipPlane    = 30f;
        }

        public void SetData(float[] raw, int frames, float fps)
        {
            _raw       = raw;
            _frames    = frames;
            _fps       = fps;
            _startTime = EditorApplication.timeSinceStartup;
            _hasData   = true;
        }

        /// <summary>Returns true while animating (call Repaint on the window).</summary>
        public bool Draw(Rect rect)
        {
            if (rect.width < 2 || rect.height < 2) return false;

            if (!_hasData)
            {
                EditorGUI.DrawRect(rect, new Color(0.13f, 0.13f, 0.13f));
                var style = new GUIStyle(GUI.skin.label)
                {
                    alignment = TextAnchor.MiddleCenter,
                    normal    = { textColor = new Color(0.4f, 0.4f, 0.4f) }
                };
                GUI.Label(rect, "preview will appear after generation", style);
                return false;
            }

            int frame = (int)((EditorApplication.timeSinceStartup - _startTime) * _fps) % _frames;

            _pru.BeginPreview(rect, GUIStyle.none);
            DrawSkeleton(frame);
            DrawGrid();
            _pru.camera.Render();
            GUI.DrawTexture(rect, _pru.EndPreview());
            return true;
        }

        private void DrawSkeleton(int f)
        {
            EnsureMat();
            _mat.SetPass(0);
            GL.PushMatrix();
            GL.Begin(GL.LINES);
            GL.Color(new Color(0.25f, 0.88f, 0.50f));
            foreach (var (p, c) in Bones)
            {
                GL.Vertex(JointPos(f, p));
                GL.Vertex(JointPos(f, c));
            }
            GL.End();
            GL.PopMatrix();
        }

        private Vector3 JointPos(int frame, int joint)
        {
            int i = frame * FEAT_DIM + POS_OFFSET + joint * 3;
            if (i + 2 >= _raw.Length) return Vector3.zero;
            return new Vector3(_raw[i], _raw[i + 1], _raw[i + 2]);
        }

        private void DrawGrid()
        {
            EnsureMat();
            _mat.SetPass(0);
            GL.Begin(GL.LINES);
            GL.Color(new Color(0.27f, 0.27f, 0.27f));
            for (int i = -4; i <= 4; i++)
            {
                GL.Vertex3(-4, 0, i); GL.Vertex3(4, 0, i);
                GL.Vertex3(i, 0, -4); GL.Vertex3(i, 0, 4);
            }
            GL.End();
        }

        private static void EnsureMat()
        {
            if (_mat) return;
            _mat = new Material(Shader.Find("Hidden/Internal-Colored"))
                       { hideFlags = HideFlags.HideAndDontSave };
            _mat.SetInt("_Cull",    0);
            _mat.SetInt("_ZWrite",  0);
            _mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            _mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        }

        public void Dispose() { _pru?.Cleanup(); _pru = null; }
    }
}