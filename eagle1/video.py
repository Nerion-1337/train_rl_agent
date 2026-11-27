"""Vidéo .mp4 (≈ 25 s) d'atterrissages réussis du pilote automatique, via gymnasium.wrappers.RecordVideo.

Lancement :  python -m eagle1.video [--seconds 25]
"""
import argparse

from gymnasium.wrappers import RecordVideo

from eagle1.core import MEDIA, load_model, make_env, play_episode

FPS = 50                    # LunarLander : 50 images par seconde


def pick_seeds(model, n_steps, start=0):
    """Choisit des graines consécutives donnant des atterrissages réussis, jusqu'à couvrir n_steps."""
    seeds, total, seed = [], 0, start
    while total < n_steps:
        result = play_episode(model, seed=seed)
        if result["success"] and result["outcome"] == "atterrissage":
            seeds.append(seed)
            total += result["n_steps"] + 1
        seed += 1
    return seeds


def make_gif(mp4, gif, seconds=10, fps=12, width=480):
    """Aperçu GIF léger (début de la vidéo) pour le README."""
    from moviepy import VideoFileClip
    from PIL import Image

    clip = VideoFileClip(str(mp4))
    frames = [Image.fromarray(clip.get_frame(t / fps)) for t in range(int(min(seconds, clip.duration) * fps))]
    size = (width, round(frames[0].height * width / frames[0].width))
    frames = [f.resize(size) for f in frames]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=25)
    args = parser.parse_args()

    model, meta = load_model()
    n_steps = int(args.seconds * FPS)
    seeds = pick_seeds(model, n_steps)
    # step_trigger + video_length : une seule vidéo de n_steps images, à cheval sur plusieurs épisodes
    env = RecordVideo(make_env("rgb_array"), video_folder=str(MEDIA / "video_tmp"), name_prefix="eagle1",
                      step_trigger=lambda step: step == 0, video_length=n_steps, disable_logger=True)
    rewards = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(int(model.predict(obs, deterministic=True)[0]))
            total += reward
            done = terminated or truncated
        rewards.append(round(float(total), 1))
    env.close()

    out = MEDIA / "eagle1_atterrissage.mp4"
    out.unlink(missing_ok=True)
    next((MEDIA / "video_tmp").glob("*.mp4")).rename(out)
    for f in (MEDIA / "video_tmp").iterdir():
        f.unlink()
    (MEDIA / "video_tmp").rmdir()
    make_gif(out, MEDIA / "eagle1_atterrissage.gif")
    print(f"{out} — {args.seconds:.0f} s, {len(seeds)} atterrissages (graines {seeds}), récompenses {rewards} "
          f"— modèle {meta['name']}")


if __name__ == "__main__":
    main()
