"""Driver script: run the finalized generative_seamless pipeline on all
remaining sample patterns with per-image style prompts."""
import subprocess
import sys

JOBS = [
    ("16040728.png",
     "seamless botanical textile pattern, watercolor batik style, crimson "
     "pink tulip flowers, olive-green striped leaves, orange and teal accent "
     "details, dark charcoal grey marbled background, plain fabric print "
     "with no text, no signature, no logo, no watermark"),
    ("2267f5f835665645ecd3301717c0f4f3.jpg",
     "seamless floral textile pattern, yellow and peach orange flowers with "
     "green leaves and curling vines, cream ivory background, flat "
     "illustrated print style, plain fabric print with no text, no "
     "signature, no logo, no watermark"),
    ("3e929f83f05300e2d326d198f60f778d.jpg",
     "seamless watercolor floral textile pattern, colorful blue purple "
     "orange and pink flowers with green leaves, delicate watercolor "
     "illustration style, peach cream background, plain fabric print with "
     "no text, no signature, no logo, no watermark"),
    ("4041013.png",
     "seamless floral textile pattern, realistic painterly white magnolia "
     "flowers with rust brown centers, sage green leaves, dusty sage teal "
     "background, plain fabric print with no text, no signature, no logo, "
     "no watermark"),
    ("4050213.png",
     "seamless folk-art botanical textile pattern, bold outlined orange red "
     "and teal flowers, block-print illustration style, cream ivory "
     "background, plain fabric print with no text, no signature, no logo, "
     "no watermark"),
    ("439db9ba41dd92e45c139dbc795e0c9b.jpg",
     "seamless ditsy floral textile pattern, small red orange daisies "
     "purple pansies and pink flowers with green leaves, cream background, "
     "plain fabric print with no text, no signature, no logo, no watermark"),
    ("5dd4c29acc87392c903217f9b0ea1864.jpg",
     "seamless flat vector floral textile pattern, pink purple and cream "
     "flowers with dotted centers, green leaves, sage green background, "
     "plain fabric print with no text, no signature, no logo, no watermark"),
]

for fname, prompt in JOBS:
    stem = fname.rsplit(".", 1)[0]
    print(f"=== {fname} ===", flush=True)
    cmd = [
        sys.executable, "generative_seamless.py",
        f"seamless/{fname}", f"output/final/{stem}_generative.png",
        "--prompt", prompt,
        "--band", "0.12", "--guidance", "30", "--feather", "0",
        "--seed", "123", "--compare", "--tile-cols", "3", "--tile-rows", "3",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! FAILED: {fname}", flush=True)
