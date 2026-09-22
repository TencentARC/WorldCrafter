# Camera and prompt guide

Each example contains `prompt.txt`, `camera.npy`, and `actions.txt`. I2V examples
also include `image.png`. The shared `negative_prompt.txt` is loaded by default.
Tokyo street includes its original `negative_prompt.txt`; select it with
`--negative-prompt-path test/T2V/02_tokyo_street/negative_prompt.txt`.
Run the commands below from the repository root.

## Choose an example

| Mode | Example | Description |
| --- | --- | --- |
| I2V | [Cat](I2V/00_cat_vac) | Default; a cat riding a moving robot vacuum |
| I2V | [Socrates](I2V/01_socrates) | Motionless painted sculptures in a stone chamber |
| T2V | [Red balloon](T2V/00_red_balloon) | Default; a balloon floating through an abandoned street |
| T2V | [Tokyo street](T2V/02_tokyo_street) | A woman walking through a neon-lit street |

## Camera inputs

Choose either the saved poses or the action description for the same example:

```bash
python inference.py --model-type fast \
  --image-path test/I2V/01_socrates/image.png \
  --prompt-path test/I2V/01_socrates/prompt.txt \
  --camera-path test/I2V/01_socrates/camera.npy
```

Replace the last argument with `--actions-file test/I2V/01_socrates/actions.txt`
to generate the poses from actions. For T2V, use `--mode t2v`, omit `--image-path`,
and select a T2V example's prompt and trajectory.

### Write actions

```text
forward1x2
yaw_left30x3
backward1
```

This generates six chunks: two forward moves, three left turns, and one backward
move. Each chunk has 33 frames. Movement values are distances; rotation values
are degrees. Use `--num-chunks` to run only the beginning of a sequence.

| Movement | Actions | Short forms |
| --- | --- | --- |
| Forward / backward | `forward1`, `backward1` | `f1`, `b1` |
| Left / right | `left1`, `right1` | `l1`, `r1` |
| Up / down | `up1`, `down1` | Same |
| Turn left / right | `yaw_left30`, `yaw_right30` | `yl30`, `yr30` |
| Look up / down | `pitch_up15`, `pitch_down15` | `pu15`, `pd15` |

The camera starts at the origin, facing +Z, with +X to the right and +Y down.
Forward/backward and left/right follow its heading on the horizontal plane;
pitch does not change movement height. Up/down follows the world vertical axis.
Yaw turns in place. Keep the total translation distance per chunk at most 5;
split longer movements into repeated actions.

Use spaces, commas, or newlines between actions, and `#` for comments. `xN`
repeats an action. `&` combines movements and rotations in one chunk, such as
`forward2&right2&yaw_left45`; translation follows the heading at the chunk's
start. `reverseN` retraces the preceding N chunks. `reverse_framesN` replays their
sampled poses in reverse frame order. Each reverse command generates N chunks.

Some examples include headers to preserve their original sampling:

| Header | Meaning |
| --- | --- |
| `@dtype float32` | Store poses in float32 instead of the default float64 |
| `@sampling smooth_turns` | Ease motion at action changes instead of using linear sampling |
| `@last_frame include` | Include the final endpoint instead of excluding it |

Keep these headers when reproducing an example. To build poses separately:

```bash
python tools/build_trajectory.py \
  --actions-file test/I2V/01_socrates/actions.txt \
  --output-dir output/socrates_camera
```

### Supply camera poses

`camera.npy` stores global camera-to-world matrices with shape `[T, 3, 4]` or
`[T, 4, 4]`, using the same right/down/forward convention. Supply one pose per
frame and 33 frames per chunk, with translations in the model's metric scale.
The inference code derives the internal camera representations; do not
pre-normalize the file separately for UCPE or RepEncoder.

## Prompt styles

### Dynamic subjects: describe following and motion

For a moving subject that should stay in view, begin with
**“A third-person ... view closely follows ...”**. This encourages subject
following; it is a prompt cue, not a tracking constraint. Describe the subject's
appearance, its movement, and how it interacts with the surroundings. Keep
nearby obstacles and background landmarks identifiable as the subject moves.

The [Cat prompt](I2V/00_cat_vac/prompt.txt) starts:

> A third-person gameplay-like camera closely follows a gray robot vacuum moving through a modern interior with reflective hardwood floors and beautiful rays of light.

It then describes the cat, the vacuum, the furniture, and how the cat balances
during movement. Adapt the opening to the subject, for example
“A third-person trailing view closely follows a cyclist ...”.
For an environment with moving water or foliage but no followed subject, use
the scene-focused style below and describe that environmental motion directly.

### Static scenes: describe space and fixed appearance

Describe the scene as a coherent environment: its layout, foreground and
background, materials, lighting, and relationships between objects. Camera
motion comes from the trajectory. Avoid adding subject movement when the scene
should remain static.

The [Socrates prompt](I2V/01_socrates/prompt.txt) identifies the people as
**static, painted sculptures** and explicitly says that all figures remain
motionless, with rigid poses and fixed garment folds. This helps distinguish
lifelike sculptures from living people. For an ordinary room or landscape,
describe its actual contents rather than calling everything a sculpture.

### Length and consistency

Use one focused English paragraph. Around **80–120 words** is a useful starting
point; dynamic subject-following prompts often need **100–130 words** to cover
both motion and environment. These are writing guidelines, not input limits.

For I2V, keep the description consistent with the input image. For T2V, describe
the subject and setting explicitly because there is no starting image. Keep
appearance and lighting consistent throughout the paragraph, and avoid cuts,
shot changes, or camera directions that compete with the supplied trajectory.
The bundled prompts preserve the wording used for their original examples.
