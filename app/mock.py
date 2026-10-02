"""MOCK=1: a deterministic, clearly fake offline corpus so the app can be demoed without the stack.

60 invented 5-second "segments" spread over the six corpus sources (I-24 Nashville highway, PIE
Toronto dashcam, neighborhood street cam, SF street cams, synthetic warehouse, indoor smart space).
Search is token/concept overlap; verification uses per-clip ground truth, and several captions are
deliberate near-misses (lexically similar, semantically wrong) so raw-search precision is < 1.
No network calls, no real video (stream_url is always null).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAXONOMY = json.loads((HERE / "taxonomy.json").read_text())

# location -> (camera ids, capture_type, parent videos)
SOURCES = {
    "nashville": (["i24_cam-1"], "traffic", ["i24_scene03_p1c2", "i24_scene07_p2c1", "i24_scene11_p1c4"]),
    "toronto": (["pie_cam-3"], "live_driving", ["pie_set01_video_0002", "pie_set03_video_0005", "pie_set05_video_0001"]),
    "neighborhood": (["neighborhood_cam-1"], "surveillance", ["neighborhood_day01_merge", "neighborhood_day02_merge"]),
    "san_francisco": (["sf_streets_cam-1", "sf_streets_cam-2", "sf_streets_cam-3", "sf_streets_cam-4"],
                      "surveillance", ["sf_streets_1_chunk_0003", "sf_streets_2_chunk_0001",
                                       "sf_streets_3_chunk_0004", "sf_streets_4_chunk_0002"]),
    "warehouse3": (["sdg_warehouse_cam-2"], "warehouse", ["sdg_warehouse_rgb_0012", "sdg_warehouse_rgb_0027"]),
    "indoor": (["smartspace_cam-1"], "surveillance", ["smartspace_cam1_0007", "smartspace_cam1_0019"]),
}

# (location, camera index, video index, segment no, objects, caption, true taxonomy ids, near-miss note)
_CLIPS = [
    # --- I-24 Nashville, overhead multi-lane highway
    ("nashville", 0, 0, 3, "truck,car", "Overhead view of a multi-lane interstate in heavy traffic. A white semi-truck with a box trailer signals and merges from the center lane into the left lane between two sedans that slow to make room.", "truck-lane-change", ""),
    ("nashville", 0, 0, 7, "truck,car", "Dense highway traffic moving slowly in all four lanes. A red pickup truck changes lanes to the right just ahead of a gray SUV; brake lights are visible across the corridor.", "truck-lane-change", ""),
    ("nashville", 0, 0, 12, "truck,car", "Steady free-flowing highway traffic seen from an elevated camera. Cars and a few tractor-trailers keep to their lanes at constant speed; no lane changes are visible.", "", "every truck stays in its lane; nobody changes lanes"),
    ("nashville", 0, 1, 2, "car", "A black sedan in the right lane brakes sharply, its brake lights flashing, and the vehicles behind it compress quickly as traffic comes to a sudden stop on the highway.", "hard-braking", ""),
    ("nashville", 0, 1, 5, "truck,car", "Congested interstate at a merge point. A box truck squeezes into the middle lane from the on-ramp while a silver car brakes gently to let it in.", "truck-lane-change", ""),
    ("nashville", 0, 1, 9, "motorcycle,truck,car", "Wide overhead shot of the highway with light traffic. A motorcycle passes a slow-moving dump truck that stays in the left lane.", "", "the truck holds its lane and traffic is light, not dense"),
    ("nashville", 0, 2, 4, "car,truck", "Stop-and-go traffic on the interstate. Several vehicles slow down abruptly and come to a stop, creating a wave of brake lights that moves backward through the lanes.", "hard-braking", ""),
    ("nashville", 0, 2, 8, "truck,car", "An 18-wheeler in the center lane drifts across the lane marking toward the right lane in dense traffic, and a car alongside it brakes to keep its distance.", "truck-lane-change", ""),
    ("nashville", 0, 2, 13, "truck,car", "Highway traffic flows smoothly under clear daylight. Cars keep a consistent gap and a tanker truck travels in the rightmost lane.", "", "no lane change and no braking; traffic is flowing"),
    ("nashville", 0, 2, 16, "car,truck", "A white van is stopped on the right shoulder with hazard lights on while highway traffic passes it at speed.", "", "the van was already parked on the shoulder; nothing stops suddenly"),
    # --- PIE Toronto, forward dashcam drives (clear weather, daytime)
    ("toronto", 0, 0, 4, "person,car", "Dashcam view while driving on a city street on a clear sunny day. A pedestrian in a dark jacket steps off the curb into the road ahead of the moving car, and the driver slows down.", "ped-into-road", ""),
    ("toronto", 0, 0, 9, "person,car", "Driving up to an intersection in daylight. A woman pushing a stroller crosses at the marked crosswalk while the car waits at the stop line.", "", "the car is stopped at the line; the crossing is in a marked crosswalk"),
    ("toronto", 0, 0, 15, "bicycle,person,car,truck", "Forward-facing camera on a busy downtown road. A cyclist rides in the bike lane alongside moving cars, passing a parked delivery van.", "cyclist-in-traffic", ""),
    ("toronto", 0, 1, 3, "car", "The car ahead brakes suddenly at a yellow light and the dashcam vehicle stops hard behind it, close to its rear bumper.", "hard-braking", ""),
    ("toronto", 0, 1, 8, "person,car", "Residential street drive on a bright afternoon. A pedestrian waits on the sidewalk next to the road as moving vehicles pass; he never steps off the curb.", "", "the pedestrian stays on the sidewalk"),
    ("toronto", 0, 1, 14, "car,person", "Daytime drive on a wide avenue. A taxi pulls over to the curb ahead and a passenger opens the rear door and gets out.", "curbside-pickup", ""),
    ("toronto", 0, 1, 20, "bicycle,car", "City driving in clear weather. A cyclist swerves around a parked car into the traffic lane right in front of the camera vehicle.", "cyclist-in-traffic", ""),
    ("toronto", 0, 2, 2, "person,car", "A group of schoolchildren crosses the road in front of the stopped car at a crosswalk on a sunny morning.", "", "the car is stopped, not moving, while the children cross"),
    ("toronto", 0, 2, 6, "person,car", "Dashcam footage on a two-lane road. A jogger suddenly runs out between parked cars into the street as the car approaches, forcing the driver to brake.", "ped-into-road,hard-braking", ""),
    ("toronto", 0, 2, 11, "bus,person,car", "Driving behind a city bus that stops at a bus stop; passengers board at the curb while traffic flows in the next lane.", "", "a bus at a bus stop, not a car at the curb"),
    ("toronto", 0, 2, 17, "person,car", "Clear daytime drive past a construction zone; orange cones narrow the lane and a worker holds a stop sign.", "", "a work zone, not the requested event"),
    ("toronto", 0, 2, 23, "bicycle,car", "Approaching an intersection in daylight, a delivery cyclist rides between two lanes of moving cars.", "cyclist-in-traffic", ""),
    # --- Neighborhood street camera (fixed, residential)
    ("neighborhood", 0, 0, 5, "car,person", "Fixed camera overlooking a residential street lined with houses. A silver sedan stops at the curb and a woman gets out of the passenger side carrying a bag.", "curbside-pickup", ""),
    ("neighborhood", 0, 0, 11, "car", "A dark SUV passes in front of the houses at moderate speed; the street is otherwise empty in daylight.", "", "no stop and no people"),
    ("neighborhood", 0, 0, 18, "car", "Two cars approach each other on the narrow residential street; one pulls toward the curb and stops briefly to let the other pass.", "", "the car stops at the curb but nobody gets in or out"),
    ("neighborhood", 0, 1, 3, "person,car", "A person walks down a driveway and gets into a parked hatchback at the curb, which then drives away.", "curbside-pickup", ""),
    ("neighborhood", 0, 1, 9, "bicycle,person,truck", "A child on a bicycle rides along the edge of the street as a pickup truck passes slowly beside them.", "cyclist-in-traffic", ""),
    ("neighborhood", 0, 1, 14, "truck,person", "A delivery van double-parks in front of a house; the driver gets out and carries a package to the door.", "curbside-pickup", ""),
    ("neighborhood", 0, 1, 21, "person,car", "Dusk on the residential street: streetlights flicker on and a person crosses the street toward a parked car while the sky is still bright.", "", "it is dusk with a bright sky, not night"),
    ("neighborhood", 0, 1, 26, "car", "A car reverses out of a driveway into the street while another vehicle brakes hard to avoid it.", "hard-braking", ""),
    # --- San Francisco street cameras
    ("san_francisco", 0, 0, 2, "person,car", "Busy San Francisco intersection with pedestrians crossing in the crosswalk while cars wait at the red light.", "", "the cars are waiting at a red light"),
    ("san_francisco", 0, 0, 6, "person,car", "A pedestrian steps off the curb mid-block and crosses in front of a moving car, which slows to avoid them.", "ped-into-road", ""),
    ("san_francisco", 0, 0, 10, "bicycle,car", "A cyclist rides up the hill in the traffic lane next to a cable car and moving cars.", "cyclist-in-traffic", ""),
    ("san_francisco", 1, 1, 3, "car,person", "A rideshare car stops at the curb with hazard lights on and two passengers climb into the back seat.", "curbside-pickup", ""),
    ("san_francisco", 1, 1, 7, "person", "A crowd of people gathers at the entrance of a store, blocking the doorway as shoppers squeeze in.", "crowded-doorway", ""),
    ("san_francisco", 1, 1, 12, "car", "Wet-looking pavement reflects the traffic lights; cars drive through the intersection under an overcast sky with no rain or snow falling.", "", "the road looks wet but no rain or snow is falling"),
    ("san_francisco", 2, 2, 4, "truck,car", "A delivery truck brakes abruptly as a car cuts in front of it at the intersection.", "hard-braking", ""),
    ("san_francisco", 2, 2, 9, "person,car", "Night-time street scene: storefronts are lit and a few pedestrians walk along the sidewalk while cars pass with headlights on.", "", "pedestrians walk along the sidewalk; nobody crosses the street"),
    ("san_francisco", 2, 2, 15, "bicycle,car", "Cyclists and scooter riders share the bike lane alongside slow-moving traffic on Market Street.", "cyclist-in-traffic", ""),
    ("san_francisco", 3, 3, 5, "person,car", "A pedestrian jaywalks across the street between moving cars, and one car brakes suddenly to let them pass.", "ped-into-road,hard-braking", ""),
    ("san_francisco", 3, 3, 11, "car,bus", "A parked car pulls away from the curb into traffic as a bus approaches.", "", "the car leaves the curb; nobody gets in or out"),
    ("san_francisco", 3, 3, 16, "person", "A group of tourists crowds the sidewalk near a cable car stop, spilling into the street.", "", "a crowd on an open sidewalk, not a corridor or doorway"),
    # --- Synthetic warehouse (ceiling / aisle cameras)
    ("warehouse3", 0, 0, 2, "person,forklift", "Ceiling camera view of a warehouse aisle. A worker in a high-visibility vest walks along the racks while a forklift carrying a pallet drives past within a few feet of him.", "person-near-forklift", ""),
    ("warehouse3", 0, 0, 6, "forklift", "A forklift moves down an empty aisle between tall shelving racks; no people are visible.", "", "no person is near the forklift"),
    ("warehouse3", 0, 0, 10, "pallet", "A pallet wrapped in plastic is left in the middle of the walkway, blocking the marked pedestrian path between racks.", "blocked-walkway", ""),
    ("warehouse3", 0, 0, 14, "person,forklift", "Two workers stand in an aisle talking while a forklift reverses toward them with its warning lights flashing.", "person-near-forklift", ""),
    ("warehouse3", 0, 0, 19, "box", "A stack of boxes has fallen into the aisle and partially obstructs the forklift lane.", "blocked-walkway", ""),
    ("warehouse3", 0, 1, 3, "person,forklift,pallet", "Wide view of the warehouse floor: workers load boxes onto a pallet near the loading dock while a forklift sits parked and idle.", "", "the forklift is parked, not moving"),
    ("warehouse3", 0, 1, 8, "person,forklift", "A person pushes a hand cart across the aisle in front of a moving forklift, which stops to let them pass.", "person-near-forklift", ""),
    ("warehouse3", 0, 1, 12, "pallet", "An empty aisle with neatly stacked pallets on both sides and a clear walkway.", "", "the walkway is clear"),
    ("warehouse3", 0, 1, 17, "person,forklift", "A forklift operator maneuvers a pallet onto a low shelf while a worker walks right behind the moving forklift in the same aisle.", "person-near-forklift", ""),
    ("warehouse3", 0, 1, 22, "box,person", "Cardboard boxes and a pallet jack are left in the aisle, narrowing the walkway; a worker steps around them.", "blocked-walkway", ""),
    # --- Indoor smart spaces
    ("indoor", 0, 0, 3, "person", "Indoor corridor camera: a group of about eight people crowds the doorway of a meeting room, waiting to enter.", "crowded-doorway", ""),
    ("indoor", 0, 0, 8, "", "An empty office corridor with fluorescent lighting; no people are present.", "", "the corridor is empty"),
    ("indoor", 0, 0, 13, "person,laptop", "A person walks alone through the hallway carrying a laptop.", "", "a single person, no crowd"),
    ("indoor", 0, 0, 19, "person", "People stream through the building lobby entrance and a cluster forms at the security gate, slowing the flow.", "crowded-doorway", ""),
    ("indoor", 0, 1, 2, "person", "A cleaning cart is parked in the middle of the corridor, blocking the walkway as a person squeezes past.", "blocked-walkway", ""),
    ("indoor", 0, 1, 6, "person", "Two people chat near the elevator doors while others pass by in the corridor.", "", "two people chatting is not a crowd"),
    ("indoor", 0, 1, 11, "person", "A crowd of attendees fills the corridor outside an auditorium after a session ends.", "crowded-doorway", ""),
    ("indoor", 0, 1, 15, "box", "Boxes are stacked in front of an emergency exit door, obstructing the hallway path.", "blocked-walkway", ""),
]

SEGMENTS: list[dict] = []
for _loc, _cam, _vid, _seg, _objs, _caption, _truth, _note in _CLIPS:
    _cams, _ctype, _videos = SOURCES[_loc]
    _video = _videos[_vid]
    SEGMENTS.append({
        "source": f"s3://mock-vss-chunks-segments/demo/{_video}_seg{_seg:03d}.mp4",
        "original_video": f"s3://mock-vss-chunks/demo/{_video}.mp4",
        "filename": f"{_video}_seg{_seg:03d}.mp4",
        "camera_id": _cams[_cam], "location": _loc, "capture_type": _ctype,
        "start_sec": float((_seg - 1) * 5), "end_sec": float(_seg * 5), "segment_number": _seg,
        "caption": _caption, "objects": [o for o in _objs.split(",") if o],
        "truth": {t for t in _truth.split(",") if t}, "note": _note,
    })
BY_SOURCE = {s["source"]: s for s in SEGMENTS}

# --------------------------------------------------------------------------- text "embeddings"

_STOP = set("a an the of in on at or and to with by for from into onto is are be it its it's this that "
            "while as up down out one two few some very just still right over under than then their them "
            "his her him he she they we you no not never next close near about other others each both all "
            "seen shows showing clip video".split())


def _stem(word: str) -> str:
    """Tiny suffix stripper so 'stopping'/'stops'/'stopped' and 'brake'/'braking' meet."""
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("es") and len(word) > 4 and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
        word = word[:-1]  # stopp -> stop, runn -> run
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]  # brake -> brak (matches braking)
    return word


_GROUPS = {
    "person": "pedestrian person people man woman worker jogger child children kid passenger tourist "
              "attendee someone schoolchildren shopper operator rider",
    "vehicle": "vehicle car sedan suv van taxi hatchback bus pickup rideshare",
    "truck": "truck semi semi-truck 18-wheeler tractor-trailer trailer tanker lorry",
    "traffic": "traffic congested congestion stop-and-go",
    "cyclist": "cyclist bicycle bike biker cycling scooter",
    "forklift": "forklift",
    "brake": "brake stop halt sudden suddenly abruptly sharply hard",
    "lane": "lane merge squeeze change swerve drift cut",
    "crowd": "crowd group cluster gather fill stream",
    "block": "block obstruct obstacle narrow fallen",
    "aisle": "aisle walkway corridor hallway path",
    "road": "road street crosswalk intersection avenue",
    "curb": "curb curbside double-park pull",
    "night": "night night-time nighttime dark",
    "weather": "rain snow wet rainy snowy",
    "door": "door doorway entrance exit gate",
    "cross": "cross jaywalk",
    "move": "moving move drive driving approach pass",
    "step": "step run walk",
    "warehouse": "warehouse rack shelving shelf",
    "highway": "highway interstate freeway",
}
_CONCEPT = {_stem(w): concept for concept, words in _GROUPS.items() for w in words.split()}


def concepts(text: str) -> set[str]:
    out = set()
    for word in re.findall(r"[a-z0-9][a-z0-9\-']*", (text or "").lower()):
        if word in _STOP:
            continue
        stem = _stem(word)
        out.add(_CONCEPT.get(stem, stem))
    return out


def _unit(*parts: str) -> float:
    """Deterministic pseudo-random number in [0, 1)."""
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16) / 0x100000000


_SEG_CONCEPTS = {s["source"]: concepts(s["caption"]) for s in SEGMENTS}
_SEG_OBJECTS = {s["source"]: concepts(" ".join(s["objects"])) for s in SEGMENTS}


def _hit(seg: dict, similarity: float) -> dict:
    hit = {k: seg[k] for k in ("source", "original_video", "filename", "camera_id", "location", "capture_type",
                               "start_sec", "end_sec", "segment_number", "caption", "objects")}
    hit["similarity"] = round(similarity, 4)
    hit["raw"] = {"mock": True, "source": seg["source"], "similarity_score": round(similarity, 4),
                  "segment_start_sec": seg["start_sec"], "segment_end_sec": seg["end_sec"],
                  "camera_id": seg["camera_id"], "location": seg["location"],
                  "capture_type": seg["capture_type"], "object_classes": ",".join(seg["objects"])}
    return hit


def search(query: str, top_k: int = 25, min_similarity: float = 0.3, location: str | None = None,
           hybrid_text_weight: float | None = None) -> list[dict]:
    """Concept-overlap "hybrid" search: caption overlap blended with detected-object overlap."""
    q = concepts(query)
    if not q:
        return []
    weight = 0.6 if hybrid_text_weight is None else max(0.0, min(1.0, float(hybrid_text_weight)))
    scored = []
    for seg in SEGMENTS:
        if location and seg["location"] != location:
            continue
        text = 0.95 * (len(q & _SEG_CONCEPTS[seg["source"]]) / len(q)) ** 2
        visual = 0.08 + 0.5 * len(q & _SEG_OBJECTS[seg["source"]]) / len(q)
        score = weight * text + (1 - weight) * visual + (_unit(seg["source"], query) - 0.5) * 0.04
        if score >= min_similarity:
            scored.append((score, seg))
    scored.sort(key=lambda pair: -pair[0])
    return [_hit(seg, score) for score, seg in scored[: max(1, int(top_k))]]


def get(source: str) -> dict | None:
    seg = BY_SOURCE.get(source)
    return _hit(seg, 0.0) if seg else None


def location_values() -> list[str]:
    return sorted(SOURCES)


# --------------------------------------------------------------------------- expansion

_EXPANSIONS = {
    "ped-into-road": ["person walking into the street in front of an approaching car",
                      "pedestrian steps off the curb as a car drives toward them",
                      "jogger runs out between parked cars into the road",
                      "pedestrian jaywalks across the street between moving cars"],
    "person-near-forklift": ["worker walking beside a moving forklift between warehouse racks",
                             "forklift drives past a person in a warehouse aisle",
                             "person pushes a cart in front of a moving forklift",
                             "forklift reverses toward workers standing in an aisle"],
    "truck-lane-change": ["semi truck merges into the next lane in heavy highway traffic",
                          "pickup truck changes lanes between cars on a congested interstate",
                          "box truck squeezes into the middle lane from the on-ramp",
                          "18-wheeler drifts across the lane marking in dense traffic"],
    "hard-braking": ["car ahead brakes suddenly and comes to a hard stop",
                     "brake lights flash as traffic stops abruptly",
                     "vehicle stops hard to avoid a collision",
                     "truck brakes abruptly at an intersection"],
    "curbside-pickup": ["car pulls over to the curb and a passenger gets out",
                        "person gets into a parked car at the curb",
                        "taxi stops at the curb to pick up passengers",
                        "rideshare car stopped curbside with passengers climbing in"],
    "cyclist-in-traffic": ["cyclist rides in the traffic lane alongside moving cars",
                           "bicycle rider passes between lanes of moving traffic",
                           "cyclist swerves around a parked car into traffic",
                           "person on a bicycle rides next to a passing truck"],
    "crowded-doorway": ["crowd of people gathered at a doorway",
                        "group of people fills a hallway corridor",
                        "people cluster at a building entrance gate",
                        "crowd blocks the entrance of a room"],
    "blocked-walkway": ["pallet left in the middle of a warehouse walkway",
                        "boxes obstruct the aisle between shelving racks",
                        "cart blocking a corridor walkway",
                        "obstacle narrowing the pedestrian path in an aisle"],
    "night-crossing": ["pedestrian crosses the street at night under streetlights",
                       "person walking across the road in the dark",
                       "night-time crosswalk with a pedestrian and headlights",
                       "pedestrian crossing a dark street"],
    "rain-or-snow": ["car driving on a wet road in heavy rain",
                     "dashcam view of driving through falling snow",
                     "rainy street with wipers on and wet pavement",
                     "snow-covered road with traffic"],
}


def taxonomy_entry(scenario: str) -> dict | None:
    """The taxonomy entry a free-text scenario refers to (exact, or strong concept overlap)."""
    norm = " ".join((scenario or "").lower().split())
    for entry in TAXONOMY:
        if norm in (entry["query"].lower(), entry["label"].lower(), entry["id"]):
            return entry
    target = concepts(scenario)
    best, best_score = None, 0.0
    for entry in TAXONOMY:
        q = concepts(entry["query"])
        score = len(q & target) / max(1, len(q | target))
        if score > best_score:
            best, best_score = entry, score
    return best if best_score >= 0.6 else None


def expand(scenario: str) -> list[str]:
    """Templated 'LLM' expansion: canned rewrites for taxonomy scenarios, synonym swaps otherwise."""
    entry = taxonomy_entry(scenario)
    if entry:
        return list(_EXPANSIONS[entry["id"]])
    swaps = {"pedestrian": "person walking", "vehicle": "car", "car": "vehicle", "road": "street",
             "street": "road", "person": "pedestrian", "people": "group of people", "cyclist": "person on a bike"}
    base = " ".join(scenario.split()).strip(" .")
    swapped = " ".join(swaps.get(w.lower(), w) for w in base.split())
    out = [swapped, f"camera view of {base}", f"{base} close to the camera", f"{base} in daylight"]
    return [q for i, q in enumerate(out) if q.lower() != base.lower() and q not in out[:i]][:4]


# --------------------------------------------------------------------------- verification

def _best_sentence(caption: str, scenario: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.;:])\s+", caption) if s.strip()]
    target = concepts(scenario)
    best = max(sentences, key=lambda s: len(concepts(s) & target)) if sentences else caption
    best = best.rstrip(".;:")
    return best if len(best) <= 170 else best[:169].rstrip() + "…"


async def judge(scenario: str, source: str) -> dict:
    """Deterministic fake 'Cosmos3-Reason' verdict from per-clip ground truth (or overlap)."""
    seg = BY_SOURCE.get(source)
    await asyncio.sleep(0.25 + 0.45 * _unit(source, scenario, "latency"))
    if seg is None:
        return {"match": None, "confidence": None, "why": "unknown mock clip", "method": "mock"}
    entry = taxonomy_entry(scenario)
    if entry:
        match = entry["id"] in seg["truth"]
    else:
        target = concepts(scenario)
        match = len(target & _SEG_CONCEPTS[source]) / max(1, len(target)) >= 0.6
    noise = _unit(source, scenario, "confidence")
    confidence = round((0.78 + 0.18 * noise) if match else (0.70 + 0.23 * noise), 2)
    evidence = _best_sentence(seg["caption"], scenario)
    if match:
        why = f"Visible: {evidence[0].lower() + evidence[1:]}."
    elif seg["note"] and entry:
        why = f"Not a match: {seg['note']}."
    else:
        why = f"Not a match: the clip shows {evidence[0].lower() + evidence[1:]}."
    return {"match": match, "confidence": confidence, "why": why, "method": "mock", "model": "mock-cosmos"}
