from tools.build_ship_dispatch_labels import build_ship_dispatch_labels


def row(token):
    return {"token": token, "timestamp": 0, "scene_token": "scene_a"}


def target(instance_token, action, confidence=1.0):
    intention = (
        "ship_entering_lock" if action == "dispatch_enter" else "ship_leaving_lock"
    )
    return {
        "instance_token": instance_token,
        "annotation_token": f"ann_{instance_token}",
        "category": "Cargo_ship",
        "dispatch_action": action,
        "ship_intention": intention,
        "assigned_berth_slot": None,
        "occlusion_state": "no_or_minor_occlusion",
        "visibility_level": "v80-100",
        "visibility_token": "4",
        "confidence": confidence,
    }


def test_ship_dispatch_labels_are_separate_from_gate_water_actions():
    labels = build_ship_dispatch_labels(
        [row("enter"), row("leave"), row("idle")],
        {
            "enter": [target("ship_1", "dispatch_enter", confidence=0.75)],
            "leave": [
                target("ship_2", "dispatch_exit", confidence=1.0),
                target("ship_3", "dispatch_exit", confidence=0.9),
            ],
        },
    )

    assert labels["enter"]["ship_dispatch_action"] == "dispatch_enter"
    assert labels["enter"]["ship_dispatch_target_count"] == 1
    assert labels["enter"]["ship_dispatch_confidence"] == 0.75
    assert labels["leave"]["ship_dispatch_action"] == "dispatch_exit"
    assert labels["leave"]["ship_dispatch_target_count"] == 2
    assert labels["leave"]["ship_dispatch_confidence"] == 0.9
    assert labels["idle"]["ship_dispatch_action"] == "hold"
    assert labels["idle"]["ship_dispatch_targets"] == []
    assert labels["idle"]["ship_dispatch_confidence"] == 1.0


def test_ship_dispatch_conflict_is_explicit_when_entering_and_leaving_mix():
    labels = build_ship_dispatch_labels(
        [row("mixed")],
        {
            "mixed": [
                target("ship_entering", "dispatch_enter"),
                target("ship_leaving", "dispatch_exit"),
            ]
        },
    )

    assert labels["mixed"]["ship_dispatch_action"] == "dispatch_conflict"
    assert labels["mixed"]["ship_dispatch_conflict"] is True
