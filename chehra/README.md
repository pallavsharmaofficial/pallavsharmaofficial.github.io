# Chehra

Real-time face reading in the browser, on the device. No upload, no server.
Module 1: **Dusq placement**. It finds the spot behind the ear lobe and guides the device there live.

Live: https://pallavsharmaofficial.github.io/chehra/

## How it works
1. **Face mesh** (MediaPipe Face Landmarker, 478 3D points, vendored in `vendor/`) builds a head frame
   (down / back axes, face width, iris-based mm scale) while the user starts facing the camera and turns.
2. **Ear-locked tracker** (`track.js`, pyramidal Lucas-Kanade + robust similarity fit) takes over
   past ~60° of turn, where the mesh drops out, so the full "look at your shoulder" profile still works.
3. **Scan**: a pose-independent patch around the target (sampled on the back/down plane) is stored as bare skin.
4. **Guide / fit check**: the live patch is compared with that scan (lighting gain, drift-tolerant matching), and the device blob
   gives offset in mm, rotation (4th-order moment, square symmetry) and size. Voice and beeps work without looking at the screen.
   Auto-captures when within tolerance for 1.2 s.

The target, rotation, size and tolerances are in Settings. **Tap to set target** is the precise calibration:
turn to the side and tap the exact spot once. It is saved on the device.

## Android
It's a PWA. Open the link in Chrome, then use menu → *Install app*. For the Play Store, wrap the same URL as a
Trusted Web Activity (Bubblewrap). No code changes are needed.

## Updating the ML assets
`tools/vendor_mediapipe.sh chehra face_landmarker/face_landmarker/float16/1/face_landmarker.task`
(reusable for any future hand/pose/face app on this site).
