"""Prompts for indoor navigation instruction generation.

Each instruction type pairs a prompt with the frames the model sees. Route
following describes the whole route from all route frames. Object Goal and
Human Demand instructions name only the destination, so their prompts see the
goal views and the arrival frame instead of the route.
"""

from dataclasses import dataclass

SYSTEM_INSTRUCTION = (
    "You are a multimodal agent that writes English indoor navigation instructions "
    "from rendered indoor trajectory images."
)

ROUTE_PROMPT = """
Describe the indoor path shown by the images.
Visual context:
Route images R1, R2, ... are RGB frames sampled along the path in chronological order.
In each route image, a thick orange marker trail is rendered on the ground to indicate the planned FUTURE path from the current position.
A red circle is rendered on the floor around the destination; the path ends at its center.
Goal images G1, G2, ..., when present, each view the same red circle from a different nearby position. They are not route steps and show no orange trail.
Use the route images for the route actions and the goal images to locate the destination among the landmarks around it. A landmark visible in the distance is not necessarily next to the destination.
The orange trail and the red circle are artificial visual guides and must not be mentioned in the generated instruction.

Ending:
The instruction should end with one goal stop sentence starting with the word "Stop" that precisely describes where to end the task.

Requirements:
- Use second-person (you).
- Perception: Mention visible landmarks or rooms. Anchor actions to a specific visible landmark, doorway, or room change. Prefer landmark-driven phrasing (pass/approach/enter/keep toward visible objects, doorways, room transitions) instead of left/right commands.
- Avoid vague phrasing: the reader must know exactly where to turn and where to go next.
- The stop sentence must describe the stopping area at the red circle through landmarks inside or right next to it.
- Do not mention the orange trail, the red circle, or any artificial visual guide.
- Do not mention meters, degrees, frame numbers, or the annotation process.

Length: 3 to 5 sentences, and 25 to 40 words total. This limit is STRICT.
Output: Output the description only. No preface, no list, no quotes.
Write the description now:
"""

OBJECT_PROMPT = """
Write an object navigation instruction that names the target object the robot must find.
Visual context:
A robot will search an indoor scene for a target and must stop at it. The images show only where the search ends.
Goal images G1, G2, ... each view the destination from a different nearby position. A red circle is rendered on the floor around the destination.
Route image R1 is the robot's own view as it arrives at the destination; it may show the red circle ahead.
The red circle is an artificial visual guide and must not be mentioned in the generated instruction.

Instruction type:
The instruction asks the robot to find one object and stop at it, for example "Find the washing machine and stop in front of it." or "Locate the black clothing rack beside the nightstand and stop next to it."

Requirements:
- Use the imperative, addressed to the robot.
- The target must be one physical object at the red circle or directly beside it, clearly visible in the goal images. Never choose open floor, a wall, a doorway, a room or an area.
- Prefer large or distinctive objects such as furniture and appliances over small items. A small item is acceptable only when it is clearly visible and stands right at the red circle.
- Name the object with a common noun. Add at most one distinguishing detail, a color, a material or one adjacent object, and only when the object could be confused with a similar one; otherwise use the plain name.
- Never describe how to get there: no route, no rooms or landmarks to pass, no turns, no left or right, no starting point.
- Do not mention the red circle, the images, meters, or the annotation process.
- End by asking the robot to stop at the object, such as in front of it, beside it, or next to it.

Length: one sentence of 6 to 18 words. This limit is STRICT.
Output: Output the instruction only. No preface, no list, no quotes.
Write the instruction now:
"""

NEED_PROMPT = """
Write a demand-driven navigation instruction: a person states a need, and the robot must find the object that satisfies it.
Visual context:
A robot will search an indoor scene for a target and must stop at it. The images show only where the search ends.
Goal images G1, G2, ... each view the destination from a different nearby position. A red circle is rendered on the floor around the destination.
Route image R1 is the robot's own view as it arrives at the destination; it may show the red circle ahead.
The red circle is an artificial visual guide and must not be mentioned in the generated instruction.

Instruction type:
The first sentence states a simple first-person need without naming the target object, such as "I am thirsty.", "I need to sit down.", "I want to rest.", "I need more light." or "I have to wash my hands.". The second sentence asks the robot to find what satisfies the need and stop there, for example "I am thirsty. Find me something cold to drink and stop next to it."

Requirements:
- Choose the target first by the rules below, then write a need that this target obviously satisfies through simple everyday reasoning, so that no other kind of object would fit the need better.
- The target must be one physical object at the red circle or directly beside it, clearly visible in the goal images. Never choose open floor, a wall, a doorway, a room or an area.
- Prefer large or distinctive objects such as furniture and appliances over small items. A small item is acceptable only when it is clearly visible and stands right at the red circle.
- Name the object with a common noun. Add at most one distinguishing detail, a color, a material or one adjacent object, and only when the object could be confused with a similar one; otherwise use the plain name.
- Never describe how to get there: no route, no rooms or landmarks to pass, no turns, no left or right, no starting point.
- Do not mention the red circle, the images, meters, or the annotation process.
- Never name the target object in the first sentence. The second sentence may describe it only by its purpose, as in "something to sit on" or "a place to rest", not by its name.

Length: two sentences of 8 to 22 words in total. This limit is STRICT.
Output: Output the instruction only. No preface, no list, no quotes.
Write the instruction now:
"""


@dataclass(frozen=True)
class InstructionType:
    prompt: str
    # Route frames shown to the model: all of them, or only the last, the arrival view.
    route_frames: str


INSTRUCTION_TYPES = {
    "vln": InstructionType(ROUTE_PROMPT, "all"),
    "object": InstructionType(OBJECT_PROMPT, "last"),
    "need": InstructionType(NEED_PROMPT, "last"),
}
