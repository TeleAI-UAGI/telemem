from eval.EvalModel import MemoryEval

data_path = "data/ZH-4O_locomo_format.json"
memory_manager = MemoryEval(data_path=data_path)
memory_manager.data = memory_manager.load_data()
memory_manager.process_all_conversation(max_workers=28)
memory_manager.build_all_graphs()
memory_manager.process_search_answer()


