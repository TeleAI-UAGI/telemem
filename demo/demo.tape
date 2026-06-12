# TeleMem terminal demo — generates docs/assets/demo.gif
# Requires: vhs (https://github.com/charmbracelet/vhs) + OPENAI_API_KEY set
# Usage: vhs demo/demo.tape

Output docs/assets/demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 900
Set Height 550
Set Theme "Dracula"
Set TypingSpeed 40ms
Set PlaybackSpeed 1.0

# --- install ---
Type "pip install telemem" Sleep 500ms Enter
Sleep 3s

# --- add memories ---
Type "python - <<'EOF'" Enter
Type "import telemem as mem0" Enter
Type "m = mem0.Memory()" Enter
Type "" Enter
Type "# two-character conversation" Enter
Type "msgs = [" Enter
Type '    {"role": "user",      "content": "Jordan, did you take the subway today?"},' Enter
Type '    {"role": "assistant", "content": "Yes, Line 2 → Civic Center, Exit A."},' Enter
Type "]" Enter
Type 'm.add(messages=msgs, user_id="Jordan")' Enter
Type 'print("✅ Memories stored")' Enter
Type "EOF" Enter
Sleep 8s

# --- search ---
Type "python - <<'EOF'" Enter
Type "import telemem as mem0" Enter
Type "m = mem0.Memory()" Enter
Type 'results = m.search("How does Jordan commute?", user_id="Jordan")' Enter
Type 'for r in results["results"]:' Enter
Type '    print("·", r["memory"])' Enter
Type "EOF" Enter
Sleep 6s

# --- MCP zero-install ---
Type "# zero-install MCP server — works with Claude Desktop, Cursor, Cline" Enter
Sleep 500ms
Type "uvx telemem --help" Enter
Sleep 4s
