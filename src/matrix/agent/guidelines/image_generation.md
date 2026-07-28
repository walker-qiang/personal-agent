## Image & Video Generation Guidelines
When calling `agnes.generate_image` or `agnes.generate_video`, follow these rules:

**What you do (creative description):**
- Translate the user's intent to English
- Describe the visual content: subject, scene, action, pose, expression, environment
- Describe the composition: camera angle, framing, depth of field, lighting
- Describe the mood and atmosphere: warm/cold, tense/calm, bright/dark
- Be specific and concrete — avoid vague terms like "beautiful" or "nice"
- Keep the prompt under 150 words, focused on visual elements

**What the code handles automatically (do NOT include):**
- Quality keywords: photorealistic, 8k, highly detailed, professional photography
- Negative prompts: no text, no watermark, no distortion, no extra limbs
- Technical specs: resolution, format, rendering engine

**Example:**
- User says "一只猫" → your prompt: "A fluffy orange tabby cat sitting on a wooden windowsill, soft morning light streaming through lace curtains, shallow depth of field focusing on the cat's green eyes, warm cozy atmosphere, dust particles dancing in the light"
- User says "老虎捕猎北极熊" → your prompt: "A Siberian tiger in mid-pounce, muscles tensed, mouth open showing sharp teeth, targeting a polar bear on a snowy Arctic ice field, dramatic overcast sky, snow particles in the air, low camera angle, intense action shot, cold blue-white color palette"

**Style guidance:**
- If the user asks for a specific style (artistic, anime, oil-painting, sketch, 3d-render, watercolor), describe it in the prompt — e.g., "anime style illustration of..."
- Default is photorealistic — no need to mention it explicitly
- For videos, use the default settings (1152x768, 121 frames, 24fps ≈ 5 seconds). Only change if the user asks for specific duration or quality.

**Video generation note:**
- Video generation is asynchronous and takes 2-3 minutes. The tool will wait for completion automatically.
- After calling the tool, show the result with: ![描述](video_url)