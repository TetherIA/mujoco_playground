# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for tetheria hand."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "tetheria_hand_tendon"
CUBE_XML = ROOT_PATH / "xmls" / "scene_mjx_cube.xml"

NQ = 16
NV = 16
NU = 7

JOINT_NAMES = [
    # index
    "right_index_mcp_flex",
    "right_index_pip",
    "right_index_dip",
    # middle
    "right_middle_mcp_flex",
    "right_middle_pip",
    "right_middle_dip",
    # ring
    "right_ring_mcp_flex",
    "right_ring_pip",
    "right_ring_dip",
    # pinky
    "right_pinky_mcp_flex",
    "right_pinky_pip",
    "right_pinky_dip",
    # thumb
    "right_thumb_cmc_abd",
    "right_thumb_cmc_flex",
    "right_thumb_mcp",
    "right_thumb_ip",
]

ACTUATOR_NAMES = [
    # index
    "right_index_A_pip",
    # middle
    "right_middle_A_pip",
    # ring
    "right_ring_A_pip",
    # pinky
    "right_pinky_A_pip",
    # thumb
    "right_thumb_A_cmc_abd",
    "right_th1_A_pip",
    "right_th2_A_pip",
]

FINGERTIP_NAMES = [
    "if_tip",
    "mf_tip",
    "rf_tip",
    "pf_tip",
    "th_tip",
]
