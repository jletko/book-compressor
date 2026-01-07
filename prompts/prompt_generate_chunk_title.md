**Role**: Creative Editor.

**Task**: Generate a localized **Chapter Header** for the text below in **{target_language}**.

**Guidelines**:
*   **Format**: "WORD {chunk_number}: TITLE" (All Uppercase).
    *   *Example*: "KAPITOLA {chunk_number}: PRÍCHOD JARI"
*   **Content**: Title should be short (2-8 words) and match the text's tone ({style_specific_instruction}).
*   **Constraint**: STRICTLY use chunk number **{chunk_number}**. Ignore source text numbering.
*   **Edge Cases**: If text is only front-matter, title it "INTRODUCTION" (localized).

**Input Text**:
{text_chunk}
