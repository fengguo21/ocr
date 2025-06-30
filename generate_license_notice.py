#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
许可证声明自动生成工具
用于为商业项目生成PaddleOCR相关的许可证声明
"""

import os
from datetime import datetime

class LicenseNoticeGenerator:
    """许可证声明生成器"""
    
    def __init__(self, project_name: str, company_name: str = ""):
        self.project_name = project_name
        self.company_name = company_name
        self.current_year = datetime.now().year
        
    def generate_license_file(self, output_dir: str = "."):
        """生成LICENSE文件"""
        license_content = f"""
{self.project_name}

本项目使用了以下开源组件：

================================================================================
PaddleOCR
================================================================================
Copyright (c) 2020 PaddlePaddle Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

项目地址: https://github.com/PaddlePaddle/PaddleOCR

================================================================================

{f"Copyright (c) {self.current_year} {self.company_name}" if self.company_name else ""}
本项目的其他部分遵循 [您的许可证] 许可证。
""".strip()
        
        license_path = os.path.join(output_dir, "LICENSE")
        with open(license_path, 'w', encoding='utf-8') as f:
            f.write(license_content)
        
        print(f"✅ LICENSE文件已生成: {license_path}")
        
    def generate_notice_file(self, output_dir: str = "."):
        """生成NOTICE文件"""
        notice_content = f"""
{self.project_name}
{f"Copyright (c) {self.current_year} {self.company_name}" if self.company_name else ""}

本项目包含以下第三方组件：

================================================================================
PaddleOCR
================================================================================
Copyright (c) 2020 PaddlePaddle Authors
Licensed under the Apache License, Version 2.0
项目地址: https://github.com/PaddlePaddle/PaddleOCR

感谢PaddlePaddle团队提供的优秀OCR解决方案！
""".strip()
        
        notice_path = os.path.join(output_dir, "NOTICE")
        with open(notice_path, 'w', encoding='utf-8') as f:
            f.write(notice_content)
            
        print(f"✅ NOTICE文件已生成: {notice_path}")
        
    def generate_readme_section(self):
        """生成README.md中的许可证部分"""
        readme_section = f"""
## 🙏 致谢

本项目基于以下优秀的开源项目：

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** - 百度飞桨OCR工具库
  - 版权所有: (c) 2020 PaddlePaddle Authors
  - 许可证: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
  - 感谢PaddlePaddle团队提供的强大OCR技术支持

## 📄 许可证

本项目的源代码遵循 [您的许可证名称] 许可证。

第三方组件的许可证信息请参见：
- [LICENSE](LICENSE) - 完整许可证文本
- [NOTICE](NOTICE) - 第三方组件声明

## ⚖️ 法律声明

本项目使用PaddleOCR技术，该技术遵循Apache 2.0许可证。
在商业使用时，请确保遵循相关开源许可证条款。
""".strip()

        print("📄 README.md许可证部分：")
        print("=" * 50)
        print(readme_section)
        print("=" * 50)
        
        return readme_section
        
    def generate_source_header(self, file_path: str = ""):
        """生成源代码文件头部声明"""
        header = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{self.project_name}{f" - {file_path}" if file_path else ""}

本模块基于PaddleOCR开发
PaddleOCR版权所有 (c) 2020 PaddlePaddle Authors
Licensed under the Apache License, Version 2.0

项目地址: https://github.com/PaddlePaddle/PaddleOCR
许可证: http://www.apache.org/licenses/LICENSE-2.0

{f"Copyright (c) {self.current_year} {self.company_name}" if self.company_name else ""}
"""
'''
        print(f"💻 源代码文件头部声明{f' ({file_path})' if file_path else ''}：")
        print("=" * 50)
        print(header)
        print("=" * 50)
        
        return header
        
    def generate_web_notice(self):
        """生成网页版权声明"""
        web_notice = f'''
<!-- 网站底部版权声明 -->
<div class="license-notice">
    <p>本服务基于 <a href="https://github.com/PaddlePaddle/PaddleOCR">PaddleOCR</a> 构建</p>
    <p>PaddleOCR © 2020 PaddlePaddle Authors, Licensed under Apache 2.0</p>
</div>

<!-- 详细的第三方组件页面 -->
<div class="third-party-licenses">
    <h3>第三方开源组件</h3>
    <div class="license-item">
        <h4>PaddleOCR</h4>
        <p><strong>版权:</strong> (c) 2020 PaddlePaddle Authors</p>
        <p><strong>许可证:</strong> Apache License 2.0</p>
        <p><strong>项目地址:</strong> <a href="https://github.com/PaddlePaddle/PaddleOCR">GitHub</a></p>
        <p><strong>用途:</strong> 光学字符识别(OCR)技术</p>
    </div>
</div>
'''
        print("🌐 网页版权声明：")
        print("=" * 50)
        print(web_notice)
        print("=" * 50)
        
        return web_notice
        
    def generate_all(self, output_dir: str = "."):
        """生成所有许可证声明文件和模板"""
        print(f"🚀 为项目 '{self.project_name}' 生成许可证声明...")
        print()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件
        self.generate_license_file(output_dir)
        self.generate_notice_file(output_dir)
        
        print()
        
        # 生成模板
        self.generate_readme_section()
        print()
        self.generate_source_header("main.py")
        print()
        self.generate_web_notice()
        
        print("\n✅ 所有许可证声明已生成完成！")
        print(f"📁 文件输出目录: {os.path.abspath(output_dir)}")

def main():
    """主函数"""
    print("📋 PaddleOCR 商业使用许可证声明生成器")
    print("=" * 50)
    
    # 用户输入
    project_name = input("请输入项目名称: ").strip()
    company_name = input("请输入公司名称 (可选): ").strip()
    output_dir = input("请输入输出目录 (默认当前目录): ").strip() or "."
    
    # 生成声明
    generator = LicenseNoticeGenerator(project_name, company_name)
    generator.generate_all(output_dir)

if __name__ == "__main__":
    main() 