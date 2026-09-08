from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

# Modell laden
model, processor = load(MODEL)
config = load_config(MODEL)

# Deine Bilder
# images = [
#     "reference_tests/old_3_0.jpg",
#     "reference_tests/old_3_1.jpg",
#     "reference_tests/new_16_0.jpg"
# ]

# Deine Bilder
# images = [
#     "reference_tests/old_5_0.jpg",
#     "reference_tests/old_5_1.jpg",
#     "reference_tests/new_22_0.jpg"
# ]

# Deine Bilder
images = [
    "reference_tests/old_1_0.jpg",
    "reference_tests/old_1_1.jpg",
    "reference_tests/new_18_0.jpg"
]

prompt = """
You are performing a person re-identification verification.

The images are provided in this exact order:

Image 1 = OLD_A
Image 2 = OLD_B
Image 3 = NEW_A

OLD_A and OLD_B are confirmed to show the SAME physical person.
They are reference images from a previously lost track.

Your task is to determine whether NEW_A shows that SAME physical person
or a DIFFERENT person.

Use BOTH OLD_A and OLD_B together as the reference appearance.


IMPORTANT CONTEXT:

The old and new tracks have already been selected as possible matches
because a previous color-histogram comparison found relatively high
color similarity between them.

Therefore, similar dominant colors and similar clothing colors are already
expected and provide only weak additional evidence for SAME.

Your task is to go BEYOND this known color similarity and search for
additional person-specific visual evidence that can confirm or reject
the proposed match.


VISUAL COMPARISON:

Compare all clearly visible person-specific characteristics, including:

- facial appearance, if clearly visible
- glasses
- hair and hairstyle
- body build and body proportions
- exact clothing design, shape and cut
- logos, prints, stripes and patterns
- trousers or shorts
- shoes
- bags and accessories
- other distinctive visible characteristics

Do NOT assign a fixed importance to one category.

Judge each characteristic according to how clearly visible, distinctive
and reliably comparable it is.

Consider the COMBINATION of the available evidence rather than making
the decision from one characteristic alone.


RELIABILITY OF VISUAL DETAILS:

Only use a characteristic when it is clearly visible and reliably
comparable between the old and new images.

If a characteristic is hidden, blurry, cropped, too small or otherwise
uncertain, treat it as UNKNOWN.

The absence of a small detail in one image does NOT mean that the detail
is actually absent. It may simply not be visible because of resolution,
viewpoint or occlusion.

Do not invent or infer details that cannot be clearly observed.


SMALL OR CHANGEABLE DETAILS:

Be careful with small or changeable characteristics such as:

- small logos
- sock details
- shoe details
- glasses
- small accessories
- minor hairstyle differences

These can provide supporting evidence, but a single such difference must
NOT determine the final identity decision.

For example, a logo visible in NEW_A but not visible in OLD_A or OLD_B
is NOT automatically a contradiction. It is only a contradiction if the
corresponding area is clearly visible in both image groups and the actual
appearance is clearly different.


DECISION RULES:

1. Do NOT classify SAME mainly because clothing colors match.
   This color similarity is already known from the histogram comparison.

2. A SAME decision should be supported by multiple compatible
   person-specific characteristics beyond the known color similarity.

3. Do NOT classify DIFFERENT because of one small or uncertain detail.

4. A DIFFERENT decision should normally require at least TWO independent,
   clearly visible and reliable person-specific contradictions.

5. Alternatively, DIFFERENT may be chosen when ONE major characteristic
   provides a very strong and unambiguous contradiction.

6. A small local detail, such as a sock logo, shoe detail, glasses or minor
   accessory difference, is NOT a major contradiction by itself.

7. If one small detail appears different but the remaining reliable
   person-specific characteristics are consistent, do not use that small
   detail alone as evidence for DIFFERENT.

8. Differences caused only by lighting, viewpoint, pose, scale or partial
   occlusion are NOT identity differences.

9. Completely IGNORE:
   - background and surroundings
   - scene content
   - camera angle itself
   - position within the image
   - distance from the camera

   These factors must NEVER be used as identity evidence.

10. Before deciding, consider BOTH the strongest reliable similarities
    and the strongest reliable contradictions across all provided images.

11. Base the final decision on the overall person-specific evidence, not
    on whichever single difference is easiest to notice.


OUTPUT:

Give ONE short sentence explaining the most decisive reliable evidence.

The explanation should reflect the overall comparison and must not base
the decision on a single small detail.

Then output exactly one final decision:

REASON: <one short sentence>
FINAL: SAME

or

REASON: <one short sentence>
FINAL: DIFFERENT
"""

formatted_prompt = apply_chat_template(
    processor,
    config,
    prompt,
    num_images=len(images)
)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=images,
    max_tokens=200,
    verbose=True
)

print("\nErgebnis:")
print(result.text.strip().upper())