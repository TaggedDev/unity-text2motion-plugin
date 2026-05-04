// Editor/SkeletonPreview.cs
using UnityEditor;
using UnityEngine;

namespace TextToMotion.Editor
{
    /// <summary>
    /// Draws an animated skeleton wireframe inside an IMGUIContainer
    /// using PreviewRenderUtility + Handles.
    /// </summary>
    public sealed class SkeletonPreview : System.IDisposable
    {
        // HumanML3D 22-joint parent→child connections
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

        public const int FEAT_DIM   = 263;
        public const int POS_OFFSET = 136;
        public const int NUM_JOINTS = 22;

        private PreviewRenderUtility _pru;
        private float[]  _raw;
        private int      _frames;
        private float    _fps      = 20f;
        private double   _startTime;
        private bool     _hasData;

        // Cached joint positions for current frame
        private readonly Vector3[] _joints = new Vector3[NUM_JOINTS];

        public SkeletonPreview() => InitPRU();

        private void InitPRU()
        {
            _pru = new PreviewRenderUtility();
            _pru.camera.transform.position = new Vector3(0f, 1.2f, -3.5f);
            _pru.camera.transform.LookAt(new Vector3(0f, 0.9f, 0f));
            _pru.camera.clearFlags          = CameraClearFlags.SolidColor;
            _pru.camera.backgroundColor     = new Color(0.13f, 0.13f, 0.13f);
            _pru.camera.nearClipPlane       = 0.05f;
            _pru.camera.farClipPlane        = 30f;
            _pru.camera.fieldOfView         = 50f;

            // One directional light
            _pru.lights[0].intensity  = 1f;
            _pru.lights[0].transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }

        public void SetData(float[] raw, int frames, float fps)
        {
            _raw        = raw;
            _frames     = frames;
            _fps        = fps > 0f ? fps : 20f;
            _startTime  = EditorApplication.timeSinceStartup;
            _hasData    = true;
        }

        /// <summary>Returns true while animating (signals window to Repaint).</summary>
        public bool Draw(Rect rect)
        {
            if (rect.width < 4 || rect.height < 4) return false;

            // ── No data placeholder ───────────────────────────────────────
            if (!_hasData)
            {
                EditorGUI.DrawRect(rect, new Color(0.13f, 0.13f, 0.13f));
                var style = new GUIStyle(GUI.skin.label)
                {
                    alignment = TextAnchor.MiddleCenter,
                    normal    = { textColor = new Color(0.4f, 0.4f, 0.4f) }
                };
                GUI.Label(rect, "Preview will appear after generation", style);
                return false;
            }

            // ── Lazy re-init if PRU was cleaned ───────────────────────────
            if (_pru == null) InitPRU();

            int frame = (int)(
                (EditorApplication.timeSinceStartup - _startTime) * _fps
            ) % Mathf.Max(_frames, 1);

            ExtractJoints(frame);

            // ── Render background + grid via PRU ──────────────────────────
            _pru.BeginPreview(rect, GUIStyle.none);
            DrawGridGL();
            _pru.camera.Render();
            Texture previewTex = _pru.EndPreview();

            // Blit camera result
            GUI.DrawTexture(rect, previewTex);

            // ── Draw skeleton lines via Handles (correct 2D→3D projection) ─
            Handles.SetCamera(rect, _pru.camera);
            Handles.color = new Color(0.25f, 0.88f, 0.50f, 1f);

            foreach (var (p, c) in Bones)
                Handles.DrawLine(_joints[p], _joints[c], 2f);

            // Joint dots
            Handles.color = new Color(1f, 0.85f, 0.2f, 1f);
            foreach (var j in _joints)
                Handles.DotHandleCap(0, j, Quaternion.identity, 0.025f, EventType.Repaint);

            return true;   // keep animating
        }

        // ── Extract world positions from raw array ────────────────────────

        private void ExtractJoints(int frame)
        {
            for (int j = 0; j < NUM_JOINTS; j++)
            {
                int i = frame * FEAT_DIM + POS_OFFSET + j * 3;
                _joints[j] = (i + 2 < _raw.Length)
                    ? new Vector3(_raw[i], _raw[i + 1], _raw[i + 2])
                    : Vector3.zero;
            }
        }

        // ── Grid drawn inside PRU (before camera.Render) ─────────────────

        private static Material _gridMat;

        private static Material GridMat()
        {
            if (_gridMat != null) return _gridMat;
            _gridMat = new Material(Shader.Find("Hidden/Internal-Colored"))
                           { hideFlags = HideFlags.HideAndDontSave };
            _gridMat.SetInt("_Cull",     0);
            _gridMat.SetInt("_ZWrite",   0);
            _gridMat.SetInt("_ZTest",    (int)UnityEngine.Rendering.CompareFunction.Always);
            _gridMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            _gridMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            return _gridMat;
        }

        private void DrawGridGL()
        {
            GridMat().SetPass(0);

            GL.PushMatrix();
            GL.LoadProjectionMatrix(_pru.camera.projectionMatrix);
            GL.modelview = _pru.camera.worldToCameraMatrix;

            GL.Begin(GL.LINES);
            GL.Color(new Color(0.27f, 0.27f, 0.27f, 0.8f));
            for (int i = -4; i <= 4; i++)
            {
                GL.Vertex3(-4f, 0f, i);  GL.Vertex3(4f, 0f, i);
                GL.Vertex3(i,   0f, -4f); GL.Vertex3(i,  0f, 4f);
            }
            GL.End();
            GL.PopMatrix();
        }

        public void Dispose()
        {
            _pru?.Cleanup();
            _pru = null;
            if (_gridMat != null) Object.DestroyImmediate(_gridMat);
            _gridMat = null;
        }
    }
}