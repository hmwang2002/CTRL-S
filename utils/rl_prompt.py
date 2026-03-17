TEXT2SVG_QUESTION_TEMPLATE_EXPLICIT = [
    """You are an expert in SVG generation. You need to generate an SVG with the following format:

<think>
   - Canvas dimensions
   - Image description
   - Step-by-step breakdown
</think>
```svg
[Modular SVG with clear comments]
```

Let's think step by step and generate the SVG based on the instruction below. Instruction: {instruction}""",

    """You are a professional SVG developer. Please generate an SVG following this structure:

<think>
   - Canvas size
   - Overall image description
   - Numbered drawing steps
</think>
```svg
[Well-organized modular SVG code]
```

Analyze the instruction carefully and generate the SVG step by step. Instruction: {instruction}""",

    """You are a skilled SVG designer. Your task is to generate an SVG in the following format:

<think>
   - Canvas dimensions (from viewBox)
   - Description of the image
   - Sequential drawing steps
</think>
```svg
[Modular SVG with module comments]
```

Let's think step by step and create the SVG for this instruction. Instruction: {instruction}""",

    """You are an SVG generation expert. Please produce an SVG using this structure:

<think>
   - Canvas: width x height
   - Image Description: [what the image depicts]
   - Drawing Steps: [numbered list of modules]
</think>
```svg
[Structured SVG with clear module organization]
```

Let's think step by step and generate the SVG based on the instruction below. Instruction: {instruction}""",

    """You are a master of SVG development. Generate an SVG with the required format:

<think>
   - Canvas dimensions
   - Image description
   - Step-by-step module breakdown
</think>
```svg
[Modular SVG code with descriptive comments]
```

Let's think step by step and create the SVG for the given instruction. Instruction: {instruction}""",

    """You are an expert SVG graphic designer. Create an SVG following this exact format:

<think>
   - Canvas size information
   - Overall description of the image
   - Detailed drawing steps (numbered)
</think>
```svg
[Well-commented modular SVG]
```

Analyze and generate the SVG step by step based on this instruction. Instruction: {instruction}""",

    """You are a professional in vector graphics generation. Please generate an SVG with this structure:

<think>
   - Canvas: [dimensions]
   - Description: [image overview]
   - Steps: [module-by-module breakdown]
</think>
```svg
[Organized SVG with module comments]
```

Let's think step by step and create the SVG for this instruction. Instruction: {instruction}""",

    """You are an SVG development specialist. Your output should follow this format:

<think>
   - Canvas dimensions (extracted from viewBox)
   - Image description
   - Drawing steps with module identification
</think>
```svg
[Modular SVG with clear structure]
```

Let's think step by step and generate the SVG based on the instruction below. Instruction: {instruction}""",
]


IMG2SVG_QUESTION_TEMPLATE_EXPLICIT = [
    """You are an expert in SVG generation. You need to generate an SVG with the following format:

<think>
   - Canvas dimensions
   - Image description
   - Step-by-step breakdown
</think>
```svg
[Modular SVG with clear comments]
```

Let's think step by step and generate the SVG based on the provided image and instruction. 
Image: <image>
Instruction: {instruction}""",

    """You are a professional SVG developer. Please analyze the image and create an SVG following this structure:

<think>
   - Canvas size
   - Overall image description
   - Numbered drawing steps
</think>
```svg
[Well-organized modular SVG code]
```

Let's think step by step and generate the SVG based on the provided image and instruction. 
Image: <image>
Instruction: {instruction}""",

    """You are a skilled SVG designer. Your task is to convert the image into an SVG with the following format:

<think>
   - Canvas dimensions (from viewBox)
   - Description of what the image depicts
   - Sequential drawing steps
</think>
```svg
[Modular SVG with module comments]
```

Let's think step by step and generate the SVG based on the provided image and instruction. 
Image: <image>
Instruction: {instruction}""",

    """You are an SVG generation expert. Please analyze the provided image and produce an SVG using this structure:

<think>
   - Canvas: width x height
   - Image Description: [detailed description]
   - Drawing Steps: [numbered module list]
</think>
```svg
[Structured SVG with clear module organization]
```

Let's think step by step and generate the SVG based on the provided image and instruction. 
Image: <image>
Instruction: {instruction}""",

    """You are a master of SVG development. Analyze the image and generate an SVG with the required format:

<think>
   - Canvas dimensions
   - Image description
   - Step-by-step module breakdown
</think>
```svg
[Modular SVG code with descriptive comments]
```

Let's think step by step and generate the SVG based on the provided image and instruction. 
Image: <image>
Instruction: {instruction}""",

    """You are an expert SVG graphic designer. Examine the image and create an SVG following this exact format:

<think>
   - Canvas size information
   - Overall description of the image
   - Detailed drawing steps (numbered)
</think>
```svg
[Well-commented modular SVG]
```

Let's think step by step and generate the SVG based on the provided image and instruction. 
Image: <image>
Instruction: {instruction}""",

    """You are a professional in vector graphics generation. Please convert the image into an SVG with this structure:

<think>
   - Canvas: [dimensions]
   - Description: [image overview]
   - Steps: [module-by-module breakdown]
</think>
```svg
[Organized SVG with module comments]
```

Let's think step by step and generate the SVG based on the provided image and instruction. 
Image: <image>
Instruction: {instruction}""",

    """You are a skilled SVG code generator. Your task is to generate an SVG with the following format:

<think>
   - Canvas dimensions
   - Image description and key elements
   - Sequential module breakdown
</think>
```svg
[Well-structured SVG with module comments]
```

Let's think step by step and generate the SVG based on the provided image and instruction. 
Image: <image>
Instruction: {instruction}""",
]
