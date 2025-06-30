import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import difflib


class IDCardInfoExtractor:
    """身份证信息提取器"""
    
    def __init__(self):
        # 身份证字段映射
        self.field_keywords = {
            '姓名': ['姓名', '名', '姓', 'name'],
            '性别': ['性别', '男', '女', 'sex', 'gender'],
            '民族': ['民族', '汉', '族', 'nation'],
            '出生': ['出生', '生日', '年', '月', '日', 'birth'],
            '住址': ['住址', '地址', '址', '住', 'address'],
            '公民身份号码': ['公民身份号码', '身份证号', '号码', '身份号', 'id'],
            '签发机关': ['签发机关', '机关', '派出所', '公安局', 'authority'],
            '有效期限': ['有效期限', '有效期', '期限', 'valid'],
        }
        
        # 身份证号码正则表达式
        self.id_number_pattern = re.compile(r'[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[1-2]\d|3[0-1])\d{3}[\dXx]')
        
        # 出生日期正则表达式
        self.birth_date_pattern = re.compile(r'(19|20)\d{2}年(0?[1-9]|1[0-2])月(0?[1-9]|[1-2]\d|3[0-1])日')
        
        # 有效期正则表达式
        self.valid_period_pattern = re.compile(r'(19|20)\d{2}\.(0?[1-9]|1[0-2])\.(0?[1-9]|[1-2]\d|3[0-1])')
        
        # 中国省份列表
        self.provinces = [
            '北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
            '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
            '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
            '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
        ]
        
        # 民族列表
        self.ethnic_groups = [
            '汉', '蒙古', '回', '藏', '维吾尔', '苗', '彝', '壮', '布依', '朝鲜',
            '满', '侗', '瑶', '白', '土家', '哈尼', '哈萨克', '傣', '黎', '傈僳',
            '佤', '畲', '高山', '拉祜', '水', '东乡', '纳西', '景颇', '柯尔克孜',
            '土', '达斡尔', '仫佬', '羌', '布朗', '撒拉', '毛南', '仡佬', '锡伯',
            '阿昌', '普米', '塔吉克', '怒', '乌孜别克', '俄罗斯', '鄂温克',
            '德昂', '保安', '裕固', '京', '塔塔尔', '独龙', '鄂伦春', '赫哲',
            '门巴', '珞巴', '基诺'
        ]

    def extract_info(self, ocr_results: List[Tuple[str, float]]) -> Dict[str, str]:
        """从OCR结果中提取身份证信息"""
        # 合并所有OCR文本
        full_text = ' '.join([text for text, _ in ocr_results])
        
        # 初始化结果字典
        info = {
            '姓名': '',
            '性别': '',
            '民族': '',
            '出生': '',
            '住址': '',
            '公民身份号码': '',
            '签发机关': '',
            '有效期限': ''
        }
        
        # 判断是身份证正面还是背面
        is_front_side = self._is_front_side(full_text, ocr_results)
        
        if is_front_side:
            # 身份证正面：提取个人信息
            info['姓名'] = self._extract_name(full_text, ocr_results)
            info['性别'] = self._extract_gender(full_text)
            info['民族'] = self._extract_ethnicity(full_text)
            info['出生'] = self._extract_birth_date(full_text)
            info['住址'] = self._extract_address(full_text, ocr_results)
            info['公民身份号码'] = self._extract_id_number(full_text)
        else:
            # 身份证背面：只提取签发机关和有效期限
            info['签发机关'] = self._extract_authority(full_text, ocr_results)
            info['有效期限'] = self._extract_valid_period(full_text)
        
        return info

    def _is_front_side(self, text: str, ocr_results: List[Tuple[str, float]]) -> bool:
        """判断是身份证正面还是背面"""
        # 检查是否包含身份证号码（强烈指示正面）
        if self.id_number_pattern.search(text):
            return True
        
        # 检查是否包含背面特有的文字
        back_side_indicators = [
            '中华人民共和国居民身份证',
            '中华人民共和国',
            '居民身份证'
        ]
        
        # 检查是否同时包含签发机关和有效期限（指示背面）
        has_authority = '签发机关' in text
        has_validity = '有效期限' in text or '有效期' in text
        
        # 如果同时包含签发机关和有效期限，很可能是背面
        if has_authority and has_validity:
            return False
            
        # 检查背面标识文字
        for indicator in back_side_indicators:
            if indicator in text:
                return False
        
        # 检查OCR结果中的个人信息字段
        personal_info_count = 0
        personal_keywords = ['姓名', '性别', '民族', '出生']
        
        for keyword in personal_keywords:
            if keyword in text:
                personal_info_count += 1
        
        # 如果包含多个个人信息字段，可能是正面
        if personal_info_count >= 2:
            return True
            
        # 默认假设是正面（除非明确判断为背面）
        return True

    def _extract_name(self, text: str, ocr_results: List[Tuple[str, float]]) -> str:
        """提取姓名"""
        # 先尝试传统的正则匹配
        patterns = [
            r'姓名\s*([^\s\d]{2,4})',
            r'名\s*([^\s\d]{2,4})',
            r'姓\s*([^\s\d]{2,4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                if self._is_valid_name(name):
                    return name
        
        # 如果正则匹配失败，尝试从OCR结果中推断
        # 通常姓名是第一个完整的中文名字
        for text_item, confidence in ocr_results:
            text_clean = text_item.strip()
            # 检查是否是可能的姓名（2-4个中文字符，高置信度）
            if (len(text_clean) >= 2 and len(text_clean) <= 4 and 
                all('\u4e00' <= char <= '\u9fff' for char in text_clean) and
                confidence > 0.9 and
                self._is_valid_name(text_clean)):
                return text_clean
        
        return ''

    def _extract_gender(self, text: str) -> str:
        """提取性别"""
        if '男' in text:
            return '男'
        elif '女' in text:
            return '女'
        return ''

    def _extract_ethnicity(self, text: str) -> str:
        """提取民族"""
        for ethnicity in self.ethnic_groups:
            if ethnicity in text:
                pattern = rf'民族\s*{ethnicity}|{ethnicity}\s*族'
                if re.search(pattern, text):
                    return ethnicity + '族' if ethnicity != '汉' else '汉族'
        return ''

    def _extract_birth_date(self, text: str) -> str:
        """提取出生日期"""
        match = self.birth_date_pattern.search(text)
        if match:
            return match.group(0)
        return ''

    def _extract_address(self, text: str, ocr_results: List[Tuple[str, float]]) -> str:
        """提取住址"""
        # 先尝试传统的正则匹配
        patterns = [
            r'住址\s*([^姓名性别民族出生公民身份号码签发机关有效期限]+)',
            r'地址\s*([^姓名性别民族出生公民身份号码签发机关有效期限]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                address = match.group(1).strip()
                if len(address) > 5:
                    return address
        
        # 如果正则匹配失败，尝试从OCR结果中推断
        # 查找包含省、市、区、县、镇、村、街道、路、号等地址关键词的文本
        address_keywords = ['省', '市', '区', '县', '镇', '村', '街', '路', '号', '巷', '里', '栋', '楼', '室']
        address_parts = []
        
        for text_item, confidence in ocr_results:
            text_clean = text_item.strip()
            # 检查是否包含地址关键词且置信度较高
            if (any(keyword in text_clean for keyword in address_keywords) and 
                confidence > 0.8 and len(text_clean) > 3):
                address_parts.append(text_clean)
        
        # 合并地址部分
        if address_parts:
            # 按文本长度排序，较长的可能是更完整的地址
            address_parts.sort(key=len, reverse=True)
            # 尝试合并相邻的地址部分
            combined_address = ''.join(address_parts)
            
            # 清理地址文本，移除不相关的字段
            unwanted_fields = ['公民身份号码', '姓名', '性别', '民族', '出生', '签发机关', '有效期限']
            for field in unwanted_fields:
                combined_address = combined_address.replace(field, '')
            
            # 清理多余的空格和特殊字符
            combined_address = re.sub(r'\s+', '', combined_address)
            
            if len(combined_address) > 8:
                return combined_address
            elif address_parts:
                # 对单个地址部分也进行清理
                cleaned_address = address_parts[0]
                for field in unwanted_fields:
                    cleaned_address = cleaned_address.replace(field, '')
                cleaned_address = re.sub(r'\s+', '', cleaned_address)
                return cleaned_address if len(cleaned_address) > 3 else address_parts[0]
        
        return ''

    def _extract_id_number(self, text: str) -> str:
        """提取身份证号码"""
        match = self.id_number_pattern.search(text)
        if match:
            id_number = match.group(0)
            if self._validate_id_number(id_number):
                return id_number
        return ''

    def _extract_authority(self, text: str, ocr_results: List[Tuple[str, float]]) -> str:
        """提取签发机关"""
        # 先尝试正则匹配
        patterns = [
            r'签发机关\s*([^有效期限]+)',
            r'机关\s*([^有效期限]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                authority = match.group(1).strip()
                # 清理可能的干扰文字
                authority = re.sub(r'有效期限.*', '', authority)
                authority = authority.strip()
                if ('公安局' in authority or '派出所' in authority) and len(authority) > 3:
                    return authority
        
        # 如果正则匹配失败，从OCR结果中查找
        for text_item, confidence in ocr_results:
            text_clean = text_item.strip()
            # 查找包含"公安局"或"派出所"的文本
            if (('公安局' in text_clean or '派出所' in text_clean) and 
                confidence > 0.8 and 
                len(text_clean) > 4 and
                '签发机关' not in text_clean):  # 避免包含字段名的文本
                return text_clean
                
        return ''

    def _extract_valid_period(self, text: str) -> str:
        """提取有效期限"""
        if '长期' in text:
            return '长期'
        
        patterns = [
            r'有效期限\s*([\d\.\-年月日至长期]+)',
            r'有效期\s*([\d\.\-年月日至长期]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return ''

    def _is_valid_name(self, name: str) -> bool:
        """验证姓名是否有效"""
        if not name or len(name) < 2 or len(name) > 4:
            return False
        if re.search(r'[\d\W]', name):
            return False
        invalid_names = ['性别', '民族', '出生', '住址', '机关', '有效', '期限']
        return name not in invalid_names

    def _validate_id_number(self, id_number: str) -> bool:
        """验证身份证号码"""
        if len(id_number) != 18:
            return False
        if not id_number[:17].isdigit():
            return False
        if id_number[17] not in '0123456789Xx':
            return False
        
        # 验证校验码
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
        
        sum_val = sum(int(id_number[i]) * weights[i] for i in range(17))
        check_code = check_codes[sum_val % 11]
        
        return id_number[17].upper() == check_code

    def post_process_info(self, info: Dict[str, str]) -> Dict[str, str]:
        """后处理提取的信息"""
        processed_info = info.copy()
        
        # 处理出生日期格式
        if processed_info['出生']:
            processed_info['出生'] = self._normalize_date(processed_info['出生'])
        
        # 处理有效期格式
        if processed_info['有效期限']:
            processed_info['有效期限'] = self._normalize_valid_period(processed_info['有效期限'])
        
        # 验证身份证号码与出生日期的一致性
        if processed_info['公民身份号码'] and processed_info['出生']:
            self._validate_id_birth_consistency(processed_info)
        
        return processed_info

    def _normalize_date(self, date_str: str) -> str:
        """规范化日期格式"""
        # 尝试不同的日期格式
        date_patterns = [
            (r'(\d{4})年(\d{1,2})月(\d{1,2})日', '%Y年%m月%d日'),
            (r'(\d{4})\.(\d{1,2})\.(\d{1,2})', '%Y.%m.%d'),
            (r'(\d{4})/(\d{1,2})/(\d{1,2})', '%Y/%m/%d'),
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d')
        ]
        
        for pattern, format_str in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    year, month, day = match.groups()
                    date_obj = datetime(int(year), int(month), int(day))
                    return date_obj.strftime('%Y年%m月%d日')
                except ValueError:
                    continue
        
        return date_str

    def _normalize_valid_period(self, period_str: str) -> str:
        """规范化有效期格式"""
        if '长期' in period_str:
            return '长期'
        
        # 查找日期范围
        date_range_pattern = r'(\d{4}[\.\-/]\d{1,2}[\.\-/]\d{1,2})\s*[至\-到]\s*(\d{4}[\.\-/]\d{1,2}[\.\-/]\d{1,2}|长期)'
        match = re.search(date_range_pattern, period_str)
        if match:
            start_date, end_date = match.groups()
            start_date = self._normalize_date(start_date)
            if end_date == '长期':
                return f"{start_date}至长期"
            else:
                end_date = self._normalize_date(end_date)
                return f"{start_date}至{end_date}"
        
        return period_str

    def _validate_id_birth_consistency(self, info: Dict[str, str]) -> None:
        """验证身份证号码与出生日期的一致性"""
        id_number = info['公民身份号码']
        birth_date = info['出生']
        
        if len(id_number) == 18 and birth_date:
            # 从身份证号码提取出生日期
            id_birth = id_number[6:14]  # YYYYMMDD
            try:
                id_birth_obj = datetime.strptime(id_birth, '%Y%m%d')
                id_birth_str = id_birth_obj.strftime('%Y年%m月%d日')
                
                # 比较两个日期
                if birth_date != id_birth_str:
                    # 如果不一致，以身份证号码为准
                    info['出生'] = id_birth_str
            except ValueError:
                pass 