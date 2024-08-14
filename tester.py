import asyncio
import os
from agent_quanta_api import AgentQuantaAPI, KnowledgeBase, AgentConfig

# Ensure you have set the ANTHROPIC_API_KEY environment variable
if "ANTHROPIC_API_KEY" not in os.environ:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

class SimpleKnowledgeBase(KnowledgeBase):
    def __init__(self, data: dict):
        self.data = data

    async def query(self, question: str) -> str:
        return self.data.get(question, "No information found")

    async def update(self, data: dict) -> None:
        self.data.update(data)

async def main():
    api = AgentQuantaAPI()

    # Create knowledge bases
    tech_kb = api.create_knowledge_base(SimpleKnowledgeBase, {
        "Smart Infrastructure": "Integration of IoT devices for efficient management of city resources.",
        "Green Energy": "Renewable energy sources like solar and wind power for sustainable cities.",
        "Smart Transportation": "Electric vehicles, intelligent traffic systems, and public transit optimization."
    })
    
    env_kb = api.create_knowledge_base(SimpleKnowledgeBase, {
        "Waste Management": "Advanced recycling and composting systems to minimize landfill waste.",
        "Urban Greenery": "Integrating parks, green roofs, and vertical gardens to improve air quality.",
        "Water Conservation": "Smart water management systems to reduce waste and ensure clean water supply."
    })

    urban_kb = api.create_knowledge_base(SimpleKnowledgeBase, {
        "Urban Planning": "Designing cities for walkability, mixed-use developments, and community spaces.",
        "Smart Buildings": "Energy-efficient structures with automated systems for climate control and security.",
        "Public Spaces": "Creating inclusive, accessible spaces that promote community interaction."
    })

    # Create agents
    tech_expert = api.create_agent("TechExpert", "Smart City Technology Specialist", tech_kb)
    env_expert = api.create_agent("EnvExpert", "Environmental Sustainability Expert", env_kb)
    urban_planner = api.create_agent("UrbanPlanner", "Urban Development Specialist", urban_kb)

    # Create tools
    def cost_estimator(project: str) -> str:
        # Simulated cost estimation
        return f"Estimated cost for '{project}': $50-100 million [Simulated estimate]"

    def impact_assessment(proposal: str) -> str:
        # Simulated impact assessment
        return f"Environmental impact of '{proposal}': Moderate positive impact [Simulated assessment]"

    cost_tool = api.create_tool("cost_estimator", cost_estimator, "Estimates project costs")
    impact_tool = api.create_tool("impact_assessment", impact_assessment, "Assesses environmental impact")

    tech_expert.add_tool(cost_tool)
    env_expert.add_tool(impact_tool)
    urban_planner.add_tool(cost_tool)
    urban_planner.add_tool(impact_tool)

    # Create manager
    manager_kb = api.create_knowledge_base(SimpleKnowledgeBase, {
        "Project Management": "Coordinating multiple aspects of urban development projects.",
        "Stakeholder Engagement": "Involving community members, businesses, and government in planning."
    })
    manager = api.create_manager("ProjectDirector", "Smart City Project Director", manager_kb)
    manager.add_agent(tech_expert)
    manager.add_agent(env_expert)
    manager.add_agent(urban_planner)

    # Create quanta
    quanta = api.create_quanta("Sustainable Smart City Planning Team")
    quanta.set_manager(manager)

    # Create and execute task
    task = api.create_task(
        "Design Sustainable Smart City",
        "Develop a comprehensive plan for a sustainable smart city. Consider technological infrastructure, environmental sustainability, and urban planning aspects. Provide recommendations for key features, potential challenges, and implementation strategies."
    )

    results = await quanta.execute_task(task)

    # Print results
    print("\n--- Sustainable Smart City Project Proposal ---\n")
    for agent, result in results['agent_results'].items():
        print(f"{agent} Report:")
        print(result['llm_response'])
        print("\n" + "-"*50 + "\n")

    print("Project Director's Overview:")
    print(results['overview'])

if __name__ == "__main__":
    asyncio.run(main())