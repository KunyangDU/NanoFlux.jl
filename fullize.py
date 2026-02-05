import os

def merge_julia_project(src_path, output_name="full_project_code.md"):
    # 允许的后缀名
    extensions = ('.jl', '.toml') 
    
    with open(output_name, 'w', encoding='utf-8') as f_out:
        f_out.write(f"# Project Source Code Summary\n\n")
        
        # os.walk 会自动进入 algorithm, control 等所有子文件夹
        for root, dirs, files in os.walk(src_path):
            # 排除隐藏文件夹
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith(extensions):
                    full_path = os.path.join(root, file)
                    # 获取相对路径，例如 src/algorithm/core.jl
                    rel_path = os.path.relpath(full_path, os.getcwd())
                    
                    print(f"正在合并: {rel_path}")
                    
                    # 写入 Markdown 格式，这样 Gemini 看得最舒服
                    f_out.write(f"## File: {rel_path}\n")
                    f_out.write("```julia\n")
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f_in:
                            f_out.write(f_in.read())
                    except Exception as e:
                        f_out.write(f"# Error reading file: {e}\n")
                    
                    f_out.write("\n```\n\n---\n\n")

if __name__ == "__main__":
    # 如果你在 src 文件夹外运行，填 'src'
    # 如果你已经在 src 文件夹内运行，填 '.'
    merge_julia_project('src') 
    print("\n合并成功！请将 full_project_code.md 发送给 Gemini。")