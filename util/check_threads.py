import threading

# 列出所有目前正在運行的線程
def list_active_threads():
    threads = threading.enumerate()  # 列出所有當前的線程
    print(f"當前運行的線程數量: {len(threads)}")
    for thread in threads:
        print(f"線程名稱: {thread.name}, 守護模式: {thread.daemon}")

list_active_threads()
