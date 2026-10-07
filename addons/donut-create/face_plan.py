"""Validate the face-plan JSON accepted by Create; no ComfyUI runtime logic."""
import json

ORDINALS = ["1st", "2nd", "3rd", *[f"{index}th" for index in range(4, 17)]]


def face_prompts(value):
    if not isinstance(value, str) or len(value) > 1024 * 1024:
        raise ValueError("Face descriptions must be a JSON array of at most 16 characters.")
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate face description field.")
            result[key] = item
        return result
    try:
        rows = json.loads(value, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Face descriptions must be a JSON array.") from error
    if not isinstance(rows, list) or len(rows) > 16:
        raise ValueError("Face descriptions must be a JSON array of at most 16 characters.")
    if len(rows) == 1:
        raise ValueError("A character face plan needs at least two descriptions. Use the single face prompt control for one character.")
    names = set()
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"character", "position", "face_prompt"}
                or not isinstance(row["character"], str) or not row["character"].strip() or len(row["character"]) > 200
                or not isinstance(row["position"], (str, dict))
                or not isinstance(row["face_prompt"], str) or not row["face_prompt"].strip() or len(row["face_prompt"]) > 8000):
            raise ValueError("Each face description needs a character, resolved position and face-only prompt.")
        name = row["character"].strip().casefold()
        if name in names:
            raise ValueError("Face descriptions must identify distinct characters.")
        names.add(name)
    if rows:
        positions = [row["position"] for row in rows]
        if isinstance(positions[0], dict):
            labels = {row["character"] for row in rows}
            for row in rows:
                position = row["position"]
                if not isinstance(position, dict) or set(position) != {"left_of", "above"}:
                    raise ValueError("Relative face positions require left_of and above arrays without mixing position formats.")
                for refs in position.values():
                    if (not isinstance(refs, list) or len(refs) > 15 or any(not isinstance(ref, str) for ref in refs)
                            or len(set(refs)) != len(refs) or any(ref not in labels or ref == row["character"] for ref in refs)):
                        raise ValueError("Relative face positions must reference distinct existing characters, without self references.")
            for axis in ("left_of", "above"):
                graph = {row["character"]: row["position"][axis] for row in rows}
                visiting, visited = set(), set()
                def visit(label):
                    if label in visiting:
                        raise ValueError("Relative face positions cannot contain cycles on either axis.")
                    if label in visited:
                        return
                    visiting.add(label)
                    for other in graph[label]:
                        visit(other)
                    visiting.remove(label); visited.add(label)
                for label in graph:
                    visit(label)
        else:
            family = ["left", "center", "right"] if positions[0] in {"left", "center", "right"} else ORDINALS
            if any(not isinstance(position, str) or position not in family for position in positions) or any(
                    family.index(left) >= family.index(right) for left, right in zip(positions, positions[1:])):
                raise ValueError("Face positions must use one strictly increasing left-to-right order.")
    return rows

