using System.IO;
using UnityEditor;
using UnityEngine;

namespace TextToMotion
{
    /// <summary>
    /// Converts a denormalised HumanML3D feature array [frames × 263]
    /// into a Unity AnimationClip and saves it as a .anim asset.
    ///
    /// Layout assumed (MDM convention):
    ///   [0]       root angular velocity Y  (rad/frame)
    ///   [1–2]     root velocity XZ         (m/frame)
    ///   [3]       root height Y            (m)
    ///   [4–135]   22 joints × 6D rotation  (Zhou et al. 2019)
    ///   [136–201] 22 joints × 3D position  (world space)
    ///   [202–262] velocities / contacts    (ignored)
    ///
    /// Bone names must match transform names in your rig.
    /// </summary>
    public static class HumanML3DToClip
    {
        private const int FEAT_DIM  = 263;
        private const int NUM_JOINTS = 22;
        private const int ROT_OFF   = 4;           // start of 6D rotations
        private const int POS_OFF   = 4 + 22 * 6; // = 136, start of positions

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

        public static AnimationClip Build(float[] raw, int frames, float fps, string savePath)
        {
            var clip = new AnimationClip { frameRate = fps, name = "GeneratedMotion" };
            AnimationUtility.SetAnimationClipSettings(clip,
                new AnimationClipSettings { loopTime = false });

            float dt = 1f / fps;

            // ── Root trajectory (integrate velocity) ──────────────────────
            var cpx = new AnimationCurve(); var cpy = new AnimationCurve(); var cpz = new AnimationCurve();
            var crx = new AnimationCurve(); var cry = new AnimationCurve();
            var crz = new AnimationCurve(); var crw = new AnimationCurve();

            Vector3 pos = Vector3.zero;
            float   rotY = 0f;

            for (int f = 0; f < frames; f++)
            {
                int   off = f * FEAT_DIM;
                float t   = f * dt;
                rotY += raw[off];
                var rot = Quaternion.Euler(0f, rotY * Mathf.Rad2Deg, 0f);
                pos   += rot * new Vector3(raw[off + 1], 0f, raw[off + 2]);
                pos.y  = raw[off + 3];

                cpx.AddKey(t, pos.x); cpy.AddKey(t, pos.y); cpz.AddKey(t, pos.z);
                crx.AddKey(t, rot.x); cry.AddKey(t, rot.y); crz.AddKey(t, rot.z); crw.AddKey(t, rot.w);
            }

            clip.SetCurve(JointNames[0], typeof(Transform), "localPosition.x", cpx);
            clip.SetCurve(JointNames[0], typeof(Transform), "localPosition.y", cpy);
            clip.SetCurve(JointNames[0], typeof(Transform), "localPosition.z", cpz);
            clip.SetCurve(JointNames[0], typeof(Transform), "localRotation.x", crx);
            clip.SetCurve(JointNames[0], typeof(Transform), "localRotation.y", cry);
            clip.SetCurve(JointNames[0], typeof(Transform), "localRotation.z", crz);
            clip.SetCurve(JointNames[0], typeof(Transform), "localRotation.w", crw);

            // ── Per-joint rotations (6D → quaternion) ─────────────────────
            for (int j = 1; j < NUM_JOINTS; j++)
            {
                var rx = new AnimationCurve(); var ry = new AnimationCurve();
                var rz = new AnimationCurve(); var rw = new AnimationCurve();

                for (int f = 0; f < frames; f++)
                {
                    float t   = f * dt;
                    int   off = f * FEAT_DIM + ROT_OFF + j * 6;
                    var a = new Vector3(raw[off],     raw[off + 1], raw[off + 2]);
                    var b = new Vector3(raw[off + 3], raw[off + 4], raw[off + 5]);
                    var q = Rot6D(a, b);
                    rx.AddKey(t, q.x); ry.AddKey(t, q.y); rz.AddKey(t, q.z); rw.AddKey(t, q.w);
                }

                clip.SetCurve(JointNames[j], typeof(Transform), "localRotation.x", rx);
                clip.SetCurve(JointNames[j], typeof(Transform), "localRotation.y", ry);
                clip.SetCurve(JointNames[j], typeof(Transform), "localRotation.z", rz);
                clip.SetCurve(JointNames[j], typeof(Transform), "localRotation.w", rw);
            }

            // ── Save ──────────────────────────────────────────────────────
            string dir = Path.GetDirectoryName(savePath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            AssetDatabase.CreateAsset(clip, savePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            return clip;
        }

        // Zhou et al. 2019 – 6D rotation → Quaternion
        private static Quaternion Rot6D(Vector3 a, Vector3 b)
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
    }
}