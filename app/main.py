from .agent.agent import ask_agent

def show_cli_banner():
    print(r"""
╔════════════════════════════════════════════════════════╗
║   🧠 PROJECT C - Memory's been working out. 💪          ║
╠════════════════════════════════════════════════════════╣
║  Semantic AI Agent for YouTube Wisdom & Gym Knowledge ║
║  Powered by ChromaDB · PostgreSQL · Ollama (Local LLM)║
║  Transcribes, Embeds, and Recalls From Real Talk.     ║
╚════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    show_cli_banner()
    print("🎤 Welcome to Project C")
    while True:
        query = input("\n🧠 Ask Project C anything (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        answer = ask_agent(query)
        print(f"\n🗣️ {answer}\n")
