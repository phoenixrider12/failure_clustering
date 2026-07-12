"""All prompts used in the paper, organised by case study and pipeline stage.

Keeping every prompt in one module makes the method reproducible and easy to
audit: the exact chain-of-thought reasoning prompts (Stage 1), ensemble
clustering prompts (Stage 2), assignment prompts (Stage 3), and runtime-monitor
prompts live here verbatim.

Case-study keys used throughout the codebase:
    "driving"      -- Nexar dashcam autonomous-driving crashes
    "navigation"   -- LB-WayPtNav vision-based indoor navigation collisions
    "manipulation" -- REFLECT household-kitchen robot-manipulation failures
"""

from __future__ import annotations

CASE_STUDIES = ("driving", "navigation", "manipulation")


# =========================================================================== #
# STAGE 1 -- Failure reasoning (VLM chain-of-thought)
# =========================================================================== #
# Driving and navigation take a fixed prompt over an image sequence.
REASONING_PROMPTS = {
    "driving": (
        "Describe the trajectory of a car from the sequence of images it observed along "
        "its path, knowing that it undergoes a collision. After that, provide the visual semantic "
        "reason behind its failure in brief. Pay attention to the surrounding object, other vehicles, "
        "and environment conditions. You must provide your answer in the following format -- "
        "trajectory : <trajectory_description> \n failure_reason : <semantic_failure_reason>, "
        "where <trajectory_description> is the description of its trajectory and "
        "<semantic_failure_reason> is the semantic reason behind failure."
    ),
    "navigation": (
        "Provide a description of the trajectory of a robot from the sequence of images it "
        "observed along its path, knowing that it collides in the last image. After that, provide the "
        "visual semantic reasons behind its failure in brief. Pay attention to the surrounding objects. "
        "You must provide your answer in the following format -- "
        "trajectory : <trajectory_description> \n failure_reason : <semantic_failure_reason>, "
        "where <trajectory_description> is the description of its trajectory and "
        "<semantic_failure_reason> is the semantic reason behind failure."
    ),
}

# Manipulation is task-conditioned: the prompt is filled per rollout with the
# task name, success condition and the robot's planned action sequence.
MANIPULATION_REASONING_TEMPLATE = """
A household robot failed to perform the kitchen task: '{task_name}'. The robot was supposed to perform the task the same way a human would do it in the given environment conditions. It is known that the robot failed and did some mistake. The success condition for this task is: '{success_condition}'.
The robot's plan of action was: {action_sequence}
From the image frames from robot's front camera before and after the failure of the robot's task execution and the action plan sequence, first understand what the robot did during the rollout. Pay attention to the objects it is interacting with, other objects present in the scene affecting the task, the task at hand, actions it had to take, and the mistakes it made. Then check what was the reason behind failure. Provide the earliest cause of failure.
You must provide your answer in the following format:
trajectory: <trajectory_description>
failure_reason: <semantic_failure_reason>
where <trajectory_description> is the description of its actions and <semantic_failure_reason> is the reason behind failure.
Report the exact failure reason by telling the fault in plan, or the mistake in execution, or the environmental factor causing the failure, without any additional commentary.
"""


def reasoning_prompt(case_study: str, **fields) -> str:
    """Return the Stage-1 reasoning prompt for a case study.

    For "manipulation", pass ``task_name``, ``success_condition`` and
    ``action_sequence`` to fill the task-conditioned template.
    """
    if case_study == "manipulation":
        return MANIPULATION_REASONING_TEMPLATE.format(**fields)
    if case_study in REASONING_PROMPTS:
        return REASONING_PROMPTS[case_study]
    raise KeyError(f"Unknown case study: {case_study!r}")


# =========================================================================== #
# STAGE 2 -- Taxonomy discovery
# =========================================================================== #
# Ensemble clustering prompts. Each prompt asks an LLM to independently propose
# a taxonomy; the ensemble is later aggregated (:mod:`aggregate_clusters`).
# The free-text failure reasons are appended to each prompt by the caller.
CLUSTERING_PROMPTS = {
    "driving": [
        "These are semantic failure reasons for different trajectories of a car. Your job is to analyze all of them "
        "and come up with clusters of different semantic failure reasons. Generate cluster centers based on the types "
        "of visual semantic failures present so that these reasons can be assigned to those clusters. Return the cluster "
        "names and the list of characteristics, keywords which belong to each cluster. Make sure to include long tail/rare "
        "clusters. Report the occurance frequency of each cluster.",

        "You are a domain expert in automotive collision analysis. Given a list of semantic failure reasons for car "
        "trajectories that resulted in crashes, perform the following steps:\n"
        "1. Identify and define distinct clusters of semantic failure types, covering both common incidents and "
        "long-tail/rare scenarios.\n"
        "2. For each cluster, provide:\n"
        "   • cluster_name: a concise, descriptive label\n"
        "   • keywords: a list of characteristic terms or phrases\n"
        "   • frequency: the count or percentage of occurrences in the input\n"
        "   • failure modes: a list of specific failure modes or examples\n"
        "3. Assign each failure reason to its corresponding cluster.\n"
        "4. Output the final result as a JSON array of objects with keys 'cluster_name', 'keywords', and 'frequency'.",

        "You are an expert in automotive semantic failure classification. Given a list of trajectory failure reasons "
        "that resulted in car crashes, perform the following:\n"
        "1. Identify distinct clusters of semantic failure types, including both frequent and long-tail/rare cases.\n"
        "2. For each cluster, define:\n"
        "   • cluster_name: concise label\n"
        "   • keywords: list of representative terms\n"
        "   • count: number of occurrences\n"
        "   • failure modes: specific examples\n"
        "3. Assign each failure reason to one of the clusters.",

        "You are an AI-driven taxonomy engineer for car collision analysis. Given semantic failure descriptions of "
        "trajectories that ended in crashes:\n"
        "- Group descriptions into semantically coherent clusters (include rare edge-cases).\n"
        "- For each cluster, provide:\n"
        "  • Name (short label)\n"
        "  • Key characteristics (list of keywords)\n"
        "  • Example descriptions (up to 3 representative samples)\n"
        "  • Frequency (%) of total",
    ],
    "navigation": [
        "These are semantic failure reasons of a robot navigating indoors based on images that fails due to collision. "
        "Generate cluster centers based on the types of visual semantic failures present so that these reasons can be "
        "assigned to those clusters. Return the cluster names and the list of characteristics, keywords which belong to "
        "each cluster. Make sure to include long tail/rare clusters. Report the occurance frequency of each cluster.",

        "You are an expert in robotic vision failure analysis. Below is a list of semantic failure reasons for an indoor "
        "robot navigation system that leads to collisions. Your tasks:\n"
        "  1. Identify distinct cluster centers representing each type of visual semantic failure.\n"
        "  2. Assign each failure reason to the appropriate cluster.\n"
        "  3. Include long-tail/rare clusters as separate entries.\n"
        "  4. For each cluster, report:\n"
        "     • cluster_name\n"
        "     • defining keywords or traits\n"
        "     • occurrence_frequency\n"
        "     • example descriptions (up to 3)\n"
        "  5. Present the output as a JSON array of objects with fields 'cluster_name', 'keywords', and 'frequency'.",

        "Act as a taxonomy engineer analyzing semantic failure reasons of an indoor vision-based robot that collides. "
        "Given the following descriptions, perform these steps:\n"
        "  • Group reasons into clusters based on shared semantic features.\n"
        "  • Capture both common patterns and rare/long-tail failure types.\n"
        "  • For each cluster, provide:\n"
        "      – name (a concise label)\n"
        "      – terms (list of characteristic keywords)\n"
        "      – count (number of examples in that cluster)\n"
        "      – failure modes",

        "You are a domain expert in robotic vision failure analysis. Given a list of semantic failure reasons for an "
        "indoor navigation robot that lead to collisions, perform the following steps:\n"
        "1. Identify and define distinct clusters of semantic failure types, including both common and long-tail/rare cases.\n"
        "2. For each cluster, provide:\n"
        "   • cluster_name: a concise, descriptive label\n"
        "   • keywords: list of characteristic terms or phrases\n"
        "   • frequency: count or percentage of occurrences in the input\n"
        "   • failure modes: list of specific failure modes or examples\n"
        "3. Assign each failure description to its appropriate cluster.",
    ],
    "manipulation": [
        "These are failure reasons of a household robot that fails while performing different kitchen tasks. Propose "
        "distinct groups so that these reasons can be assigned to those clusters, and these clusters represent failure "
        "modes of this robot. Return the cluster names and the list of characteristics keywords representing each "
        "cluster. Report the occurrence frequency of each cluster.",

        "You are given a list of failure reasons describing why a household robot failed while performing various "
        "kitchen tasks. Some reasons could be due to robot's mistakes, some due to environment around, some could be "
        "due to its planner, etc. Your task is to cluster these failure reasons into distinct and meaningful groups.\n"
        "For each cluster, provide:\n"
        "1. Cluster Name - a concise and descriptive label for the cluster.\n"
        "2. Characteristic Keywords - key terms that summarize the main patterns or themes in the cluster.\n"
        "3. Occurrence Frequency - how many failure reasons belong to this cluster.",

        "Act as a taxonomy engineer analyzing failure reasons of a household robot operating in kitchen environments "
        "and failing to perform the task. Given the following failure reasons, perform these steps:\n"
        "  • Group reasons into clusters based on shared features of kitchen task failures.\n"
        "  • Capture both common patterns and rare/long-tail failure types in kitchen operation tasks.\n"
        "  • For each cluster, provide:\n"
        "      – name (a concise label describing the failure type)\n"
        "      – terms (list of characteristic keywords defining the failure type)\n"
        "      – count (number of examples in that cluster)\n"
        "  • Avoid overlapping clusters and return unique clusters only.",

        "You are a domain expert in household robot's failure analysis. Given a list of failure reasons for a kitchen "
        "robot performing kitchen tasks, perform the following steps:\n"
        "   1. Identify and define distinct clusters of kitchen task failure types.\n"
        "   2. For each cluster, provide:\n"
        "       • cluster_name: a concise, descriptive label for the kitchen failure type\n"
        "       • keywords: list of characteristic terms or phrases related to failure type\n"
        "       • frequency: count or percentage of occurrences in the input\n"
        "   3. Avoid overlapping clusters and return unique clusters only.",
    ],
}

# Intro paragraph for the ensemble-aggregation step (Stage 2). The individual
# candidate taxonomy reports are appended by the caller.
AGGREGATION_INTRO = {
    "driving": (
        "You are given different reports on the failure clusters of car collisions. "
        "Your job is to aggregate these reports, and consolidate them into a single, coherent taxonomy "
        "with non-overlapping clusters. Here are the reports:\n\n"
    ),
    "navigation": (
        "You are given different reports on the failure clusters of a robot navigating indoors based on images. "
        "Your job is to aggregate these reports, and consolidate them into a single, coherent taxonomy "
        "with non-overlapping clusters. Here are the reports:\n\n"
    ),
    "manipulation": (
        "You are given different reports on the failure clusters of a household robot performing kitchen tasks. "
        "Your job is to aggregate these reports, and consolidate them into a single, coherent taxonomy "
        "with non-overlapping clusters. Here are the reports:\n\n"
    ),
}

# Stage-2 JSON conversion. The structured-output schema (schemas.ClustersResponse)
# enforces the format, so the prompt only needs to state the task.
CONVERT_TO_JSON_INSTRUCTION = (
    "Convert the following taxonomy description into a structured list of clusters. "
    "For each cluster extract its name, any occurrence/frequency information, a list of "
    "characteristic keywords, and any additional descriptive notes. Here is the text:\n\n"
)


# =========================================================================== #
# STAGE 3 -- Trajectory assignment
# =========================================================================== #
def assignment_prompt(cluster_options: list[str], case_study: str = "manipulation") -> str:
    """Build the Stage-3 assignment prompt listing the taxonomy clusters.

    ``cluster_options`` is a list of ``"<name> : <keywords> — <notes>"`` strings.
    The trajectory's ``trajectory`` and ``failure_reason`` are appended per item
    by the caller (they are left as ``{trajectory}`` / ``{failure_reason}``
    markers filled with ``str.replace`` to avoid brace-escaping issues).
    """
    subject = {
        "driving": "car",
        "navigation": "indoor navigation robot",
        "manipulation": "kitchen-robot",
    }.get(case_study, "robot")

    options_block = "\n".join(f"- {opt}" for opt in cluster_options)
    return (
        f"You are given a list of failure clusters for a {subject}, which represent the failure modes of the system.\n"
        f"Here are the clusters:\n{options_block}\n\n"
        "Your task is to classify the failure reason and trajectory description of a rollout into one or more of these "
        'clusters that best fit. If the rollout does not fit any of the clusters, assign it to "Other".\n'
        "Return a comma-separated list of cluster names (or a single cluster name) that best fit the rollout.\n\n"
        "Here is the trajectory description and failure reason:\n"
        "Trajectory description: {trajectory}\n"
        "Failure reason: {failure_reason}\n"
    )


ASSIGNMENT_SYSTEM_PROMPT = (
    "You are a classifier that classifies system failures into a predefined list of failure modes."
)


# =========================================================================== #
# Runtime monitoring
# =========================================================================== #
# For each case study we keep two prompts:
#   "taxonomy"  -- the proposed monitor, seeded with the discovered failure modes
#   "generic"   -- the ablation monitor with no taxonomy (SAFE/UNSAFE only)
#
# The taxonomy blocks below embed the failure modes discovered by Stage 2 for
# each system; replace them if you re-run taxonomy discovery on new data.
MONITOR_PROMPTS = {
    "driving": {
        "taxonomy": """
You are a runtime failure monitor for the ego car navigating outdoors.
Your task is to analyze a sequence of forward-facing camera images ending at the car's current position, and determine whether the car is:
a) SAFE — confidently following a collision-free trajectory, or
b) UNSAFE — likely to collide with another vehicle or an object

Operational Context:
- The car must keep on moving without any physical collisions.
- You are provided with a time-ordered sequence of image frames at 3 frames per second, ending at the car's current position.
- Any collision is considered a failure.

## Reasoning Procedure:
1. Trajectory Inference: From the image sequence, estimate the car's likely near-future motion.
2. Scene Understanding: Examine the latest image for road layout, obstacles, other vehicles, pedestrians, and environmental conditions. Pay extra attention to objects within or near the ego vehicle's predicted path, occluded road users, and adverse visibility.
3. Collision Risk Assessment: Determine if a collision is likely, prioritizing risks that intersect the ego path.
4. Semantic Failure Matching: You are given a list of Known Semantic Failure Reasons of this car (below). If the current scenario aligns with a known failure reason, mark it unsafe and return that exact name. If it does not match but a collision is likely, briefly describe the new semantic reason. Return SAFE only if you are confident the car will pass safely. Err on the side of caution.

Known Semantic Failure Reasons:
1. Rear-End Collisions: Insufficient Following Distance
2. Unsafe Cut-In / Lane-Change Intrusions
3. Intersection Right-of-Way Violations
4. Lane Departure & Lateral-Clearance Errors
5. Visibility-Impaired Perception Failures
6. Pedestrian & Cyclist Detection Failures
7. Static-Obstacle & Sudden Intrusion Collisions
8. Infrastructure & Clearance Errors
9. Other Rare / Long-Tail Cases

## Output Instructions:
Return only one of the following: the name of a known semantic failure reason (exactly as written above), a brief description of a new failure type, or the word SAFE.
Do not provide explanations. Output must be a single, definitive label.
""",
        "generic": """
You are a runtime failure monitor for the ego car navigating outdoors.
Analyze a sequence of forward-facing camera images ending at the car's current position, and determine whether the car is SAFE (following a collision-free trajectory) or UNSAFE (likely to collide).
Infer the near-future trajectory, understand the scene, and assess collision risk along the ego path. Err on the side of caution when unsure.
Output must be a single word: SAFE or UNSAFE.
""",
    },
    "navigation": {
        "taxonomy": """
You are a runtime failure monitor for a vision-based autonomous robot navigating in an indoor environment.
Analyze a sequence of recent image observations, ending at the robot's current position, and determine whether the robot is:
- SAFE — confidently continuing in free space on a collision-free path, or
- UNSAFE — no free space ahead and at significant and credible risk of collision.

Operational Context:
- The robot must reach a predefined goal without any physical collisions.
- You are provided with a time-ordered sequence of image frames, each 0.5 seconds apart, ending at the robot's current location.
- The robot moves at a maximum speed of 0.6 m/s.

Evaluation Procedure:
1. Predict the robot's short-term trajectory from temporal cues.
2. Identify nearby obstacles that may intersect the predicted path.
3. Determine collision risk from visual alignment of path and obstacle.
4. Classify the risk: if it matches one of the Known Semantic Failure Reasons below, return that exact label; if a new visible risk is present, briefly describe it; otherwise return SAFE.

Known Semantic Failure Reasons:
1. Thin-Protruding Objects (chair legs, table/desk legs, thin metal frames)
2. Uniform/Featureless Surfaces (white cabinets, featureless walls, panels)
3. Narrow-Gap/Clearance Misjudgment (tight passages, misjudged turning radius)
4. Low-Height Clutter & Small Floor Obstacles (backpacks, cables, debris)
5. Box-Like Equipment & Carts (computer towers, servers, carts, pedestals)
6. Structural Edges: Door Frames, Jambs & Wall Corners
7. Bins & Waste Receptacles (trash/recycling bins and lids)
8. Transparent & Reflective Surfaces (glass doors/panels, mirrors)
9. Overhead & Ceiling Fixtures (low-hanging fixtures, piping)

## Output Instructions:
Return only one of the following: the name of a known semantic failure reason (exactly as written above), a brief description of a new failure type, or the word SAFE.
Do not provide explanations. Output must be a single, definitive label.
""",
        "generic": """
You are a runtime failure monitor for a vision-based autonomous robot navigating in an indoor environment.
Analyze a sequence of recent image observations, ending at the robot's current position, and determine whether the robot is SAFE (continuing in free space on a collision-free path) or UNSAFE (no free space ahead and at credible risk of collision).
Predict the short-term trajectory, identify nearby obstacles that may intersect the path, and assess collision risk. Err on the side of caution when unsure.
Output must be a single word: SAFE or UNSAFE.
""",
    },
}
