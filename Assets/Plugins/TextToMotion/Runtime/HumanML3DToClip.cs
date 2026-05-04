// Editor/HumanML3DToClip.cs
using System.IO;
using UnityEditor;
using UnityEngine;

namespace TextToMotion
{
    /// <summary>
    /// Converts a denormalised HumanML3D feature array [frames × 263]
    /// into a Unity AnimationClip saved as a .anim asset.
    ///
    /// HumanML3D layout (MDM convention):
    ///   [0]       root angular velocity Y  (rad/frame)
    ///   [1–2]     root velocity XZ         (m/frame)
    ///   [3]       root height Y            (m)
    ///   [4–135]   22 joints × 6D rotation  (Zhou et al. 2019)
    ///   [136–201] 22 joints × 3D position  (world space, used for preview only)
    ///   [202–262] foot contacts / velocities (ignored)
    /// </summary>
    public static class HumanML3DToClip
    {
        public const int FEAT_DIM   = 263;
        public const int NUM_JOINTS = 22;
        public const int ROT_OFF    = 4;           // start of 6D rotations
        public const int POS_OFF    = 136;         // 4 + 22*6

        // Joint names must match Transform names in your preview rig
        public static readonly string[] JointNames =
        {
            "Pelvis",
            "L_Hip",      "R_Hip",      "Spine1",
            "L_Knee",     "R_Knee",     "Spine2",
            "L_Ankle",    "R_Ankle",    "Spine3",
            "L_Foot",     "R_Foot",     "Neck",
            "L_Collar",   "R_Collar",   "Head",
            "L_Shoulder", "R_Shoulder",
            "L_Elbow",    "R_Elbow",
            "L_Wrist",    "R_Wrist"
        };

        public static AnimationClip Build(
            float[] raw, int frames, float fps, string assetPath)
        {
            // ── Ensure output folder exists in AssetDatabase ──────────────
            string dir = Path.GetDirectoryName(assetPath)?.Replace('\\', '/') ?? "Assets";
            EnsureFolder(dir);

            var clip = new AnimationClip
            {
                frameRate = fps,
                name      = Path.GetFileNameWithoutExtension(assetPath),
                legacy    = false   // Generic / Humanoid compatible
            };

            var settings = AnimationUtility.GetAnimationClipSettings(clip);
            settings.loopTime        = false;
            settings.loopBlend       = false;
            settings.keepOriginalPositionXZ = false;
            settings.keepOriginalPositionY  = false;
            AnimationUtility.SetAnimationClipSettings(clip, settings);

            float dt = 1f / fps;

            // ── Root trajectory ──────────────────────────────────────────
            var cpx = new AnimationCurve();
            var cpy = new AnimationCurve();
            var cpz = new AnimationCurve();
            var crx = new AnimationCurve();
            var cry = new AnimationCurve();
            var crz = new AnimationCurve();
            var crw = new AnimationCurve();

            Vector3 pos  = Vector3.zero;
            float   rotY = 0f;

            for (int f = 0; f < frames; f++)
            {
                int   off = f * FEAT_DIM;
                float t   = f * dt;

                rotY += raw[off];
                var   rootRot = Quaternion.Euler(0f, rotY * Mathf.Rad2Deg, 0f);
                pos  += rootRot * new Vector3(raw[off + 1], 0f, raw[off + 2]);
                pos.y = raw[off + 3];

                AddKey(cpx, t, pos.x);
                AddKey(cpy, t, pos.y);
                AddKey(cpz, t, pos.z);
                AddKey(crx, t, rootRot.x);
                AddKey(cry, t, rootRot.y);
                AddKey(crz, t, rootRot.z);
                AddKey(crw, t, rootRot.w);
            }

            SetCurveSmooth(clip, JointNames[0], "localPosition.x", cpx);
            SetCurveSmooth(clip, JointNames[0], "localPosition.y", cpy);
            SetCurveSmooth(clip, JointNames[0], "localPosition.z", cpz);
            SetCurveSmooth(clip, JointNames[0], "localRotation.x", crx);
            SetCurveSmooth(clip, JointNames[0], "localRotation.y", cry);
            SetCurveSmooth(clip, JointNames[0], "localRotation.z", crz);
            SetCurveSmooth(clip, JointNames[0], "localRotation.w", crw);

            // ── Per-joint rotations (6D → Quaternion) ────────────────────
            for (int j = 1; j < NUM_JOINTS; j++)
            {
                var rx = new AnimationCurve();
                var ry = new AnimationCurve();
                var rz = new AnimationCurve();
                var rw = new AnimationCurve();

                for (int f = 0; f < frames; f++)
                {
                    float t   = f * dt;
                    int   off = f * FEAT_DIM + ROT_OFF + j * 6;

                    if (off + 5 >= raw.Length) break;

                    var a = new Vector3(raw[off],     raw[off + 1], raw[off + 2]);
                    var b = new Vector3(raw[off + 3], raw[off + 4], raw[off + 5]);
                    var q = Rot6D(a, b);

                    // Flip sign if dot with previous frame < 0 (quaternion continuity)
                    if (f > 0)
                    {
                        int  prevOff = (f - 1) * FEAT_DIM + ROT_OFF + j * 6;
                        var  pa = new Vector3(raw[prevOff],     raw[prevOff+1], raw[prevOff+2]);
                        var  pb = new Vector3(raw[prevOff + 3], raw[prevOff+4], raw[prevOff+5]);
                        var  pq = Rot6D(pa, pb);
                        if (q.x*pq.x + q.y*pq.y + q.z*pq.z + q.w*pq.w < 0f)
                            q = new Quaternion(-q.x, -q.y, -q.z, -q.w);
                    }

                    AddKey(rx, t, q.x);
                    AddKey(ry, t, q.y);
                    AddKey(rz, t, q.z);
                    AddKey(rw, t, q.w);
                }

                SetCurveSmooth(clip, JointNames[j], "localRotation.x", rx);
                SetCurveSmooth(clip, JointNames[j], "localRotation.y", ry);
                SetCurveSmooth(clip, JointNames[j], "localRotation.z", rz);
                SetCurveSmooth(clip, JointNames[j], "localRotation.w", rw);
            }

            // ── Save asset ────────────────────────────────────────────────
            AssetDatabase.CreateAsset(clip, assetPath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            Debug.Log($"[TTM] Saved AnimationClip: {assetPath}  frames={frames}  fps={fps}");
            return clip;
        }

        // ── Helpers ───────────────────────────────────────────────────────

        // Use EditorCurveBinding + AnimationUtility for correct metadata
        private static void SetCurveSmooth(
            AnimationClip clip, string boneName, string property, AnimationCurve curve)
        {
            // Smooth tangents for rotation curves (avoid gimbal artifacts)
            for (int i = 0; i < curve.length; i++)
                AnimationUtility.SetKeyLeftTangentMode(curve,  i, AnimationUtility.TangentMode.ClampedAuto);

            var binding = new EditorCurveBinding
            {
                path         = boneName,
                type         = typeof(Transform),
                propertyName = property
            };
            AnimationUtility.SetEditorCurve(clip, binding, curve);
        }

        private static void AddKey(AnimationCurve c, float t, float v)
        {
            var kf = new Keyframe(t, v);
            c.AddKey(kf);
        }

        // Zhou et al. 2019 – 6D rotation → Quaternion
        public static Quaternion Rot6D(Vector3 a, Vector3 b)
        {
            Vector3 x = a.normalized;
            Vector3 y = (b - Vector3.Dot(b, x) * x).normalized;
            Vector3 z = Vector3.Cross(x, y);

            float m00=x.x, m01=y.x, m02=z.x;
            float m10=x.y, m11=y.y, m12=z.y;
            float m20=x.z, m21=y.z, m22=z.z;

            float tr = m00 + m11 + m22;
            float qx, qy, qz, qw;

            if (tr > 0f)
            {
                float s = 0.5f / Mathf.Sqrt(tr + 1f);
                qw=(0.25f/s); qx=(m21-m12)*s; qy=(m02-m20)*s; qz=(m10-m01)*s;
            }
            else if (m00 > m11 && m00 > m22)
            {
                float s = 2f * Mathf.Sqrt(1f + m00 - m11 - m22);
                qw=(m21-m12)/s; qx=0.25f*s; qy=(m01+m10)/s; qz=(m02+m20)/s;
            }
            else if (m11 > m22)
            {
                float s = 2f * Mathf.Sqrt(1f + m11 - m00 - m22);
                qw=(m02-m20)/s; qx=(m01+m10)/s; qy=0.25f*s; qz=(m12+m21)/s;
            }
            else
            {
                float s = 2f * Mathf.Sqrt(1f + m22 - m00 - m11);
                qw=(m10-m01)/s; qx=(m02+m20)/s; qy=(m12+m21)/s; qz=0.25f*s;
            }

            return new Quaternion(qx, qy, qz, qw);
        }

        // Creates AssetDatabase-registered folder hierarchy
        public static void EnsureFolder(string folderPath)
        {
            folderPath = folderPath.Replace('\\', '/').TrimEnd('/');
            if (AssetDatabase.IsValidFolder(folderPath)) return;

            string parent = Path.GetDirectoryName(folderPath)?.Replace('\\', '/') ?? "Assets";
            string leaf   = Path.GetFileName(folderPath);
            EnsureFolder(parent);   // recurse
            AssetDatabase.CreateFolder(parent, leaf);
        }
    }
}