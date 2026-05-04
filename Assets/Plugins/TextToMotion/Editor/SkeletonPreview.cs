// Editor/SkeletonPreview.cs
using UnityEditor;
using UnityEngine;

namespace TextToMotion.Editor
{
    public sealed class SkeletonPreview : System.IDisposable
    {
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

        private float[]  _raw;
        private int      _frames;
        private float    _fps = 20f;
        private double   _startTime;
        private bool     _hasData;

        private Texture2D _tex;
        private int       _texW, _texH;
        private Color[]   _bg;

        private static readonly Color BG_COLOR   = new Color(0.13f, 0.13f, 0.13f);
        private static readonly Color BONE_COLOR = new Color(0.25f, 0.88f, 0.50f);
        private static readonly Color DOT_COLOR  = new Color(1.00f, 0.85f, 0.20f);
        private static readonly Color GRID_COLOR = new Color(0.27f, 0.27f, 0.27f);

        public void SetData(float[] raw, int frames, float fps)
        {
            _raw       = raw;
            _frames    = frames;
            _fps       = fps > 0f ? fps : 20f;
            _startTime = EditorApplication.timeSinceStartup;
            _hasData   = true;
        }

        /// <summary>Call from IMGUIContainer.onGUIHandler. Returns true while animating.</summary>
        public bool Draw(Rect rect)
        {
            if (rect.width < 4 || rect.height < 4) return false;
            if (Event.current.type != EventType.Repaint) return _hasData;

            int w = Mathf.Max((int)rect.width,  1);
            int h = Mathf.Max((int)rect.height, 1);

            EnsureTexture(w, h);

            if (!_hasData)
            {
                // Just dark background + hint text
                FillBackground();
                _tex.Apply();
                GUI.DrawTexture(rect, _tex, ScaleMode.StretchToFill, false);

                var style = new GUIStyle(GUI.skin.label)
                {
                    alignment = TextAnchor.MiddleCenter,
                    normal    = { textColor = new Color(0.4f, 0.4f, 0.4f) }
                };
                GUI.Label(rect, "Preview will appear after generation", style);
                return false;
            }

            int frame = (int)(
                (EditorApplication.timeSinceStartup - _startTime) * _fps
            ) % Mathf.Max(_frames, 1);

            // Extract 3D joints for this frame
            var joints3d = new Vector3[NUM_JOINTS];
            for (int j = 0; j < NUM_JOINTS; j++)
            {
                int i = frame * FEAT_DIM + POS_OFFSET + j * 3;
                joints3d[j] = (i + 2 < _raw.Length)
                    ? new Vector3(_raw[i], _raw[i + 1], _raw[i + 2])
                    : Vector3.zero;
            }

            // Project to 2D (simple orthographic from side)
            var joints2d = Project(joints3d, w, h);

            // Draw
            FillBackground();
            DrawGrid(w, h);
            foreach (var (p, c) in Bones)
                DrawLine(joints2d[p], joints2d[c], BONE_COLOR);
            foreach (var pt in joints2d)
                FillCircle((int)pt.x, (int)pt.y, 3, DOT_COLOR);

            _tex.Apply();
            GUI.DrawTexture(rect, _tex, ScaleMode.StretchToFill, false);
            return true;
        }

        // ── Projection ────────────────────────────────────────────────────

        private Vector2[] Project(Vector3[] pts, int w, int h)
        {
            // Find bounding box to auto-scale
            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;
            foreach (var p in pts)
            {
                if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
            }

            float rangeX = Mathf.Max(maxX - minX, 0.01f);
            float rangeY = Mathf.Max(maxY - minY, 0.01f);
            float scale  = Mathf.Min(w / rangeX, h / rangeY) * 0.75f;

            float cx = (minX + maxX) * 0.5f;
            float cy = (minY + maxY) * 0.5f;

            var result = new Vector2[pts.Length];
            for (int i = 0; i < pts.Length; i++)
            {
                float sx = (pts[i].x - cx) * scale + w * 0.5f;
                float sy = h - ((pts[i].y - cy) * scale + h * 0.5f); // flip Y
                result[i] = new Vector2(sx, sy);
            }
            return result;
        }

        // ── Texture helpers ───────────────────────────────────────────────

        private void EnsureTexture(int w, int h)
        {
            if (_tex != null && _texW == w && _texH == h) return;

            if (_tex != null) Object.DestroyImmediate(_tex);
            _tex  = new Texture2D(w, h, TextureFormat.RGBA32, false)
                        { filterMode = FilterMode.Point };
            _texW = w;
            _texH = h;
            _bg   = new Color[w * h];
            for (int i = 0; i < _bg.Length; i++) _bg[i] = BG_COLOR;
        }

        private void FillBackground() => _tex.SetPixels(_bg);

        private void DrawGrid(int w, int h)
        {
            // Horizontal lines every ~20px
            for (int y = 0; y < h; y += 20)
                for (int x = 0; x < w; x++)
                    _tex.SetPixel(x, y, GRID_COLOR);
            // Vertical lines
            for (int x = 0; x < w; x += 20)
                for (int y = 0; y < h; y++)
                    _tex.SetPixel(x, y, GRID_COLOR);
        }

        private void DrawLine(Vector2 a, Vector2 b, Color col)
        {
            int x0 = (int)a.x, y0 = (int)a.y;
            int x1 = (int)b.x, y1 = (int)b.y;

            int dx = Mathf.Abs(x1 - x0), dy = Mathf.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1,   sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;

            for (int iter = 0; iter < 2000; iter++)
            {
                SetPixelSafe(x0, y0, col);
                // Thick line — draw neighbours
                SetPixelSafe(x0+1, y0, col);
                SetPixelSafe(x0, y0+1, col);

                if (x0 == x1 && y0 == y1) break;
                int e2 = 2 * err;
                if (e2 > -dy) { err -= dy; x0 += sx; }
                if (e2 <  dx) { err += dx; y0 += sy; }
            }
        }

        private void FillCircle(int cx, int cy, int r, Color col)
        {
            for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++)
                if (dx*dx + dy*dy <= r*r)
                    SetPixelSafe(cx + dx, cy + dy, col);
        }

        private void SetPixelSafe(int x, int y, Color col)
        {
            if (x >= 0 && x < _texW && y >= 0 && y < _texH)
                _tex.SetPixel(x, y, col);
        }

        public void Dispose()
        {
            if (_tex != null) Object.DestroyImmediate(_tex);
            _tex = null;
        }
    }
}