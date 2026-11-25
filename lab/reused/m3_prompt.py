# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Exact prompt value from mmagent/prompts.py at the pinned M3 commit.
M3_PROMPT = """You will be given a video and a set of character features. Each feature is either a face (represented by a video frame with a bounding box) or a voice (represented by one or more speech segments, each with MM:SS start and end times, and transcript content). Each feature has a unique ID enclosed in angle brackets. Some features may belong to the same character.

Your task consists of two parts:
1.	Video Description:
Generate a detailed and cohesive description of the current video clip. Use the provided feature IDs as references to characters (when applicable). Your description should cover all observable and inferable events. Each description should focus on a single atomic event or fact.
2.	High-Level Conclusions:
Generate high-level reasoning-based conclusions that go beyond surface-level observations. Use logical inference to identify character intentions, relationships, and identities. If a face and a voice feature refer to the same character, indicate it using this exact format: Equivalence: <face_x>, <voice_y>

Output Format:
Your output must be a JSON object with the following structure:

{
	"video_description": [
		"...",  // each string is one atomic event description
		"..."
	],
	"high_level_conclusions": [
		"...",  // each string is one high-level inference or identity resolution
		"Equivalence: <face_1>, <voice_2>"
	]
}

Please only return the valid JSON object, without any additional explanation or formatting."""
