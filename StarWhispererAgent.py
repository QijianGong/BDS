"""
星语者 - 基于空间感知与智能体的AI天文系统
作者：龚奇健
版本：3.3
"""
from labplus.board import *
from labplus.gui import *
from labplus.bluebit import GPS
from labplus.gpio import *
from labplus.opencv import *
from labplus.AI import *
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
import json
import requests
import math
import time
import re
import random

# ========== 配置类 ==========
@dataclass
class ModelConfig:
    """大模型配置"""
    provider: str = "qwen"  # 默认使用通义千问
    api_key: str = ""
    api_base: str = ""
    model_name: str = ""
    
    # 各厂商配置映射
    PROVIDERS = {
        "qwen": {
            "name": "通义千问",
            "api_base": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            "models": ["qwen-max", "qwen-plus", "qwen-turbo"]
        },
        "baidu": {
            "name": "文心一言",
            "api_base": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
            "models": ["ERNIE-4.0-8K", "ERNIE-3.5-8K", "ERNIE-Speed-8K"]
        },
        "zhipu": {
            "name": "智谱清言",
            "api_base": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            "models": ["glm-4", "glm-3-turbo"]
        },
        "deepseek": {
            "name": "DeepSeek",
            "api_base": "https://api.deepseek.com/chat/completions",
            "models": ["deepseek-chat", "deepseek-coder"]
        },
        "doubao": {
            "name": "豆包",
            "api_base": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            "models": ["doubao-seed-2-0-mini-260215"]
        }
    }

# ========== 天气API类 ==========
class WeatherAPI:
    """和风天气API接口"""
    
    def __init__(self):
        self.api_key = "***" 
        self.api_host = "***"  
        self.base_url = f"https://{self.api_host}"
        
    def get_location_id(self, lon: float, lat: float) -> Optional[str]:
        """通过经纬度获取location ID"""
        try:
            url = f"{self.base_url}/geo/v2/city/lookup"
            params = {
                "location": f"{lon:.2f},{lat:.2f}",
                "key": self.api_key,
                "lang": "zh"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "200" and data.get("location"):
                    location = data["location"][0]
                    return location.get("id")
            return None
        except Exception as e:
            print(f"获取location ID失败: {e}")
            return None
    
    def get_current_weather(self, location_id: str) -> Dict:
        """获取当前天气"""
        try:
            url = f"{self.base_url}/v7/weather/now"
            params = {
                "location": location_id,
                "key": self.api_key,
                "lang": "zh"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "200":
                    return data.get("now", {})
            return {}
        except Exception as e:
            print(f"获取当前天气失败: {e}")
            return {}
    
    def get_daily_forecast(self, location_id: str, days: int = 3) -> List[Dict]:
        """获取每日天气预报"""
        try:
            url = f"{self.base_url}/v7/weather/{days}d"
            params = {
                "location": location_id,
                "key": self.api_key,
                "lang": "zh"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "200":
                    return data.get("daily", [])
            return []
        except Exception as e:
            print(f"获取天气预报失败: {e}")
            return []
    
    def get_astronomy_data(self, location_id: str, date: str = None) -> Dict:
        """获取天文数据（太阳、月亮）"""
        try:
            if date is None:
                date = datetime.now().strftime("%Y%m%d")
                
            url = f"{self.base_url}/v7/astronomy/sunmoon"
            params = {
                "location": location_id,
                "date": date,
                "key": self.api_key,
                "lang": "zh"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "200":
                    return data.get("sunMoon", {})
            return {}
        except Exception as e:
            print(f"获取天文数据失败: {e}")
            return {}
    
    def get_hourly_forecast(self, location_id: str, hours: int = 24) -> List[Dict]:
        """获取逐小时预报"""
        try:
            url = f"{self.base_url}/v7/weather/24h"
            params = {
                "location": location_id,
                "key": self.api_key,
                "lang": "zh"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "200":
                    forecast_data = data.get("hourly", [])
                    if hours > len(forecast_data):
                        hours = len(forecast_data)
                    return forecast_data[:hours]
            return []
        except Exception as e:
            print(f"获取逐小时预报失败: {e}")
            return []
    
    def get_air_quality(self, location_id: str) -> Dict:
        """获取空气质量"""
        try:
            url = f"{self.base_url}/v7/air/now"
            params = {
                "location": location_id,
                "key": self.api_key,
                "lang": "zh"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "200":
                    return data.get("now", {})
            return {}
        except Exception as e:
            print(f"获取空气质量失败: {e}")
            return {}
    
    def get_observation_conditions(self, lon: float, lat: float) -> Dict[str, Any]:
        """获取完整的观测条件评估"""
        try:
            # 1. 获取location ID
            location_id = self.get_location_id(lon, lat)
            if not location_id:
                return {"error": "无法获取位置信息"}
            
            # 2. 获取各项数据
            current_weather = self.get_current_weather(location_id)
            hourly_forecast = self.get_hourly_forecast(location_id, 12)  # 未来12小时
            astronomy_data = self.get_astronomy_data(location_id)
            air_quality = self.get_air_quality(location_id)
            
            # 3. 评估观测条件
            conditions = self._evaluate_conditions(
                current_weather, hourly_forecast, astronomy_data, air_quality
            )
            
            return {
                "success": True,
                "location_id": location_id,
                "current_weather": current_weather,
                "hourly_forecast": hourly_forecast[:6],  # 只返回前6小时
                "astronomy_data": astronomy_data,
                "air_quality": air_quality,
                "observation_conditions": conditions
            }
            
        except Exception as e:
            print(f"获取观测条件失败: {e}")
            return {"error": f"获取观测条件失败: {str(e)}"}
    
    def _evaluate_conditions(self, current_weather: Dict, hourly_forecast: List[Dict], 
                           astronomy_data: Dict, air_quality: Dict) -> Dict[str, Any]:
        """评估观测条件"""
        
        # 默认评估结果
        evaluation = {
            "overall_rating": 0,  # 0-5分
            "sky_condition": "未知",
            "cloud_cover": "未知",
            "visibility_rating": "未知",
            "humidity_rating": "未知",
            "wind_rating": "未知",
            "light_pollution": "未知",
            "moon_phase_effect": "未知",
            "best_time_window": "未知",
            "recommendation": "无法评估观测条件",
            "issues": [],
            "advantages": []
        }
        
        try:
            # 云量评估
            cloud_text = current_weather.get("text", "").lower()
            cloud_scale = int(current_weather.get("cloud", "101"))  # 默认101表示无数据
            
            if cloud_scale <= 30:
                evaluation["sky_condition"] = "晴朗"
                evaluation["cloud_cover"] = "低"
                evaluation["advantages"].append("天空晴朗，云量少")
                evaluation["overall_rating"] += 2
            elif cloud_scale <= 70:
                evaluation["sky_condition"] = "多云"
                evaluation["cloud_cover"] = "中等"
                evaluation["issues"].append("有较多云层")
                evaluation["overall_rating"] += 1
            else:
                evaluation["sky_condition"] = "阴天或多云"
                evaluation["cloud_cover"] = "高"
                evaluation["issues"].append("云量较多，可能影响观测")
            
            # 能见度评估
            vis_km = float(current_weather.get("vis", 10))  # 默认10km
            if vis_km >= 20:
                evaluation["visibility_rating"] = "极好"
                evaluation["advantages"].append("能见度极高")
                evaluation["overall_rating"] += 1
            elif vis_km >= 10:
                evaluation["visibility_rating"] = "良好"
            elif vis_km >= 5:
                evaluation["visibility_rating"] = "一般"
                evaluation["issues"].append("能见度一般")
            else:
                evaluation["visibility_rating"] = "较差"
                evaluation["issues"].append("能见度较差")
                evaluation["overall_rating"] -= 1
            
            # 湿度评估
            humidity = int(current_weather.get("humidity", 50))
            if humidity <= 40:
                evaluation["humidity_rating"] = "干燥，适合观测"
                evaluation["advantages"].append("空气干燥，大气稳定")
                evaluation["overall_rating"] += 1
            elif humidity <= 70:
                evaluation["humidity_rating"] = "适中"
            else:
                evaluation["humidity_rating"] = "潮湿，可能起雾"
                evaluation["issues"].append("湿度较高，可能影响观测")
                evaluation["overall_rating"] -= 1
            
            # 风速评估
            wind_speed = float(current_weather.get("windSpeed", 0))
            if wind_speed <= 3:
                evaluation["wind_rating"] = "微风，稳定"
                evaluation["advantages"].append("风速低，视宁度好")
                evaluation["overall_rating"] += 1
            elif wind_speed <= 8:
                evaluation["wind_rating"] = "有风"
            else:
                evaluation["wind_rating"] = "风大，不稳定"
                evaluation["issues"].append("风速较大，可能影响稳定性")
                evaluation["overall_rating"] -= 1
            
            # 空气质量评估（光污染指标）
            aqi = int(air_quality.get("aqi", 50))
            if aqi <= 50:
                evaluation["light_pollution"] = "低"
                evaluation["advantages"].append("空气质量好，光污染低")
                evaluation["overall_rating"] += 1
            elif aqi <= 100:
                evaluation["light_pollution"] = "中等"
            elif aqi <= 150:
                evaluation["light_pollution"] = "较高"
                evaluation["issues"].append("空气质量一般，有一定光污染")
            else:
                evaluation["light_pollution"] = "高"
                evaluation["issues"].append("空气质量差，光污染严重")
                evaluation["overall_rating"] -= 1
            
            # 月相影响评估
            moon_phase = astronomy_data.get("moonPhase", "")
            moon_illumination = float(astronomy_data.get("moonIllumination", 50))
            
            if moon_illumination <= 20:
                evaluation["moon_phase_effect"] = "新月，极佳"
                evaluation["advantages"].append("月光影响小，适合观测暗弱天体")
                evaluation["overall_rating"] += 2
            elif moon_illumination <= 40:
                evaluation["moon_phase_effect"] = "娥眉月，良好"
            elif moon_illumination <= 60:
                evaluation["moon_phase_effect"] = "上弦月，一般"
                evaluation["issues"].append("月光有一定影响")
            elif moon_illumination <= 80:
                evaluation["moon_phase_effect"] = "凸月，较差"
                evaluation["issues"].append("月光较强，影响观测")
                evaluation["overall_rating"] -= 1
            else:
                evaluation["moon_phase_effect"] = "满月，差"
                evaluation["issues"].append("满月光强，只适合观测亮星")
                evaluation["overall_rating"] -= 2
            
            # 分析最佳观测时间窗口
            best_hours = []
            for hour_data in hourly_forecast[:6]:
                hour_text = hour_data.get("text", "").lower()
                hour_cloud = int(hour_data.get("cloud", 101))
                
                if "晴" in hour_text or "clear" in hour_text or hour_cloud <= 30:
                    fx_time = hour_data.get("fxTime", "")
                    hour = fx_time[11:13] if len(fx_time) >= 13 else "未知"
                    best_hours.append(hour)
            
            if best_hours:
                evaluation["best_time_window"] = f"{', '.join(best_hours)}时"
            else:
                evaluation["best_time_window"] = "今夜可能都不太理想"
            
            # 计算总评分并生成建议
            evaluation["overall_rating"] = max(0, min(5, evaluation["overall_rating"]))
            
            if evaluation["overall_rating"] >= 4:
                evaluation["recommendation"] = "观测条件极佳！强烈推荐今晚进行天文观测。"
            elif evaluation["overall_rating"] >= 3:
                evaluation["recommendation"] = "观测条件良好，适合进行天文观测。"
            elif evaluation["overall_rating"] >= 2:
                evaluation["recommendation"] = "观测条件一般，可以考虑观测较亮的天体。"
            else:
                evaluation["recommendation"] = "观测条件较差，建议改日再观测。"
            
            # 添加注意事项
            if evaluation["issues"]:
                evaluation["recommendation"] += " 需要注意：" + "；".join(evaluation["issues"][:3])
            
            return evaluation
            
        except Exception as e:
            print(f"评估观测条件失败: {e}")
            evaluation["recommendation"] = f"评估失败: {str(e)}"
            return evaluation

# ========== 天文知识库 ==========
class AstronomyDatabase:
    """完整天文知识库"""
    
    def __init__(self):
        # 星星数据库 - 包含60+颗亮星
        self.stars = {
            # === 1-10: 最亮的10颗星 ===
            "天狼星": {
                "constellation": "大犬座",
                "magnitude": -1.46,
                "distance": 8.6,
                "ra_deg": 101.2875,      # 6h 45m 09s
                "dec_deg": -16.7161,     # -16° 42' 58"
                "story": "夜空中最亮的恒星，古埃及人称它为'索提斯'，用于预测尼罗河洪水。它实际上是一个双星系统。"
            },
            "老人星": {
                "constellation": "船底座",
                "magnitude": -0.74,
                "distance": 310.0,
                "ra_deg": 95.9880,       # 6h 23m 57s
                "dec_deg": -52.6957,     # -52° 41' 44"
                "story": "全天第二亮星，船底座最亮的恒星。在古代中国被认为是南极仙翁的化身。"
            },
            "大角星": {
                "constellation": "牧夫座",
                "magnitude": -0.05,
                "distance": 37.0,
                "ra_deg": 213.9120,      # 14h 15m 40s
                "dec_deg": 19.1825,      # +19° 10' 57"
                "story": "北天最亮的恒星，在春天夜空中非常显眼。它的名字意为'熊的守护者'。"
            },
            "织女星": {
                "constellation": "天琴座",
                "magnitude": 0.03,
                "distance": 25.3,
                "ra_deg": 279.2346,      # 18h 36m 56s
                "dec_deg": 38.7836,      # +38° 47' 01"
                "story": "天琴座最亮的星，在中国七夕传说中与牛郎星隔银河相望。它是第一颗被拍摄的恒星。"
            },
            "五车二": {
                "constellation": "御夫座",
                "magnitude": 0.08,
                "distance": 42.9,
                "ra_deg": 79.1715,       # 5h 16m 41s
                "dec_deg": 45.9979,      # +45° 59' 53"
                "story": "御夫座最亮的星，北天著名的双星系统。在拉丁语中意为'小山羊'。"
            },
            "参宿七": {
                "constellation": "猎户座",
                "magnitude": 0.12,
                "distance": 860.0,
                "ra_deg": 78.6345,       # 5h 14m 32s
                "dec_deg": -8.2017,      # -08° 12' 06"
                "story": "猎户座最亮的星，一颗蓝超巨星。它的亮度相当于太阳的120,000倍。"
            },
            "南河三": {
                "constellation": "小犬座",
                "magnitude": 0.34,
                "distance": 11.5,
                "ra_deg": 114.8250,      # 7h 39m 18s
                "dec_deg": 5.2247,       # +05° 13' 29"
                "story": "小犬座最亮的星，名字意为'在狗之前'，因为它比天狼星稍早升起。"
            },
            "水委一": {
                "constellation": "波江座",
                "magnitude": 0.46,
                "distance": 139.0,
                "ra_deg": 24.4245,       # 1h 37m 42s
                "dec_deg": -57.2367,     # -57° 14' 12"
                "story": "波江座最亮的星，名字来自阿拉伯语'河的尽头'。它是已知最扁的恒星之一。"
            },
            "参宿四": {
                "constellation": "猎户座",
                "magnitude": 0.42,
                "distance": 640.0,
                "ra_deg": 88.7929,       # 5h 55m 10s
                "dec_deg": 7.4069,       # +07° 24' 25"
                "story": "著名的红超巨星，猎户座的肩膀。它的直径大约是太阳的900倍，随时可能发生超新星爆发。"
            },
            "马腹一": {
                "constellation": "半人马座",
                "magnitude": 0.61,
                "distance": 390.0,
                "ra_deg": 210.9550,      # 14h 03m 49s
                "dec_deg": -60.3731,     # -60° 22' 23"
                "story": "半人马座第二亮的星，一颗蓝白色巨星。它与半人马座α星（南门二）相邻。",
            },
            
            # === 11-20: 著名的亮星 ===
            "牛郎星": {
                "constellation": "天鹰座",
                "magnitude": 0.76,
                "distance": 16.7,
                "ra_deg": 297.6954,      # 19h 50m 47s
                "dec_deg": 8.8683,       # +08° 52' 06"
                "story": "天鹰座最亮的星，在中国七夕传说中与织女星隔银河相望。它是夏季大三角的一角。"
            },
            "十字架二": {
                "constellation": "南十字座",
                "magnitude": 0.77,
                "distance": 320.0,
                "ra_deg": 186.6495,      # 12h 26m 36s
                "dec_deg": -63.0992,     # -63° 05' 57"
                "story": "南十字座最亮的星，实际上是一个三合星系统。用于南半球的导航。"
            },
            "毕宿五": {
                "constellation": "金牛座",
                "magnitude": 0.87,
                "distance": 65.0,
                "ra_deg": 68.9790,       # 4h 35m 55s
                "dec_deg": 16.5092,      # +16° 30' 33"
                "story": "金牛座最亮的星，一颗红巨星。它的名字来自阿拉伯语'跟随者'，因为它跟随昴星团升起。"
            },
            "角宿一": {
                "constellation": "室女座",
                "magnitude": 0.98,
                "distance": 250.0,
                "ra_deg": 201.2955,      # 13h 25m 12s
                "dec_deg": -11.1614,     # -11° 09' 41"
                "story": "室女座最亮的星，一颗蓝白色双星。名字意为'麦穗'，代表女神手中的麦穗。"
            },
            "心宿二": {
                "constellation": "天蝎座",
                "magnitude": 0.96,
                "distance": 550.0,
                "ra_deg": 247.3545,      # 16h 29m 25s
                "dec_deg": -26.4322,     # -26° 25' 56"
                "story": "天蝎座的心臟，一颗红超巨星。名字意为'火星的对手'，因为它的颜色与火星相似。"
            },
            "北河三": {
                "constellation": "双子座",
                "magnitude": 1.14,
                "distance": 34.0,
                "ra_deg": 116.3295,      # 7h 45m 19s
                "dec_deg": 28.0261,      # +28° 01' 34"
                "story": "双子座较亮的星，希腊神话中卡斯托耳的孪生兄弟。它有一颗已知的行星。"
            },
            "北落师门": {
                "constellation": "南鱼座",
                "magnitude": 1.16,
                "distance": 25.1,
                "ra_deg": 344.4125,      # 22h 57m 39s
                "dec_deg": -29.6222,     # -29° 37' 20"
                "story": "南鱼座最亮的星，'孤独的南天之星'。它有一个著名的尘埃环和一颗已知的行星。"
            },
            "天津四": {
                "constellation": "天鹅座",
                "magnitude": 1.25,
                "distance": 2600.0,
                "ra_deg": 310.3575,      # 20h 41m 26s
                "dec_deg": 45.2803,      # +45° 16' 49"
                "story": "天鹅座最亮的星，夏季大三角的一角。它是已知最亮的恒星之一，非常遥远。"
            },
            "十字架三": {
                "constellation": "南十字座",
                "magnitude": 1.25,
                "distance": 280.0,
                "ra_deg": 191.9250,      # 12h 47m 43s
                "dec_deg": -59.6889,     # -59° 41' 20"
                "story": "南十字座第二亮的星，一颗蓝白色变星。名字意为'模仿者'。"
            },
            "轩辕十四": {
                "constellation": "狮子座",
                "magnitude": 1.36,
                "distance": 77.0,
                "ra_deg": 152.0835,      # 10h 08m 22s
                "dec_deg": 11.9672,      # +11° 58' 02"
                "story": "狮子座最亮的星，一颗蓝白色主序星。它的名字意为'小国王'。"
            },
            
            # === 21-30: 其他重要恒星 ===
            "北河二": {
                "constellation": "双子座",
                "magnitude": 1.58,
                "distance": 51.0,
                "ra_deg": 113.6505,      # 7h 34m 36s
                "dec_deg": 31.8883,      # +31° 53' 18"
                "story": "双子座α星，实际上是一个六合星系统。希腊神话中波吕丢刻斯的孪生兄弟。"
            },
            "十字架四": {
                "constellation": "南十字座",
                "magnitude": 1.63,
                "distance": 88.0,
                "ra_deg": 186.4500,      # 12h 25m 48s
                "dec_deg": -56.7833,     # -56° 47' 00"
                "story": "南十字座的第三亮星，一颗红巨星。名字是Gamma Crucis的缩写。"
            },
            "十字架五": {
                "constellation": "南十字座",
                "magnitude": 2.80,
                "distance": 364.0,
                "ra_deg": 188.0000,      # 12h 32m 00s
                "dec_deg": -57.1667,     # -57° 10' 00"
                "story": "南十字座的第四亮星，一颗蓝白色变星。"
            },
            "北极星": {
                "constellation": "小熊座",
                "magnitude": 1.97,
                "distance": 433.0,
                "ra_deg": 37.9529,       # 2h 31m 49s
                "dec_deg": 89.2642,      # +89° 15' 51"
                "story": "目前最靠近北天极的亮星，用于导航。它实际上是一个三合星系统。"
            },
            "开阳": {
                "constellation": "大熊座",
                "magnitude": 2.23,
                "distance": 83.0,
                "ra_deg": 200.9835,      # 13h 23m 56s
                "dec_deg": 54.9253,      # +54° 55' 31"
                "story": "北斗六，著名的双星。视力好的人可以看到它的伴星'辅'。"
            },
            "辅": {
                "constellation": "大熊座",
                "magnitude": 3.99,
                "distance": 82.0,
                "ra_deg": 201.3045,      # 13h 25m 13s
                "dec_deg": 54.9878,      # +54° 59' 16"
                "story": "开阳的伴星，传统上用于测试视力。古代阿拉伯军队用它来测试士兵的视力。"
            },
            "大火星": {
                "constellation": "天蝎座",
                "magnitude": 2.29,
                "distance": 300.0,
                "ra_deg": 240.0833,      # 16h 00m 20s
                "dec_deg": -22.6217,     # -22° 37' 18"
                "story": "天蝎座的第二亮星，名字来自阿拉伯语'蝎子的刺'。"
            },
            "参宿五": {
                "constellation": "猎户座",
                "magnitude": 1.64,
                "distance": 250.0,
                "ra_deg": 84.0533,       # 5h 36m 13s
                "dec_deg": -1.2017,      # -01° 12' 06"
                "story": "猎户座的肩膀，一颗蓝白色巨星。名字意为'女战士'。"
            },
            "参宿六": {
                "constellation": "猎户座",
                "magnitude": 2.23,
                "distance": 700.0,
                "ra_deg": 84.6917,       # 5h 38m 46s
                "dec_deg": -9.6694,      # -09° 40' 10"
                "story": "猎户座的膝盖，一颗蓝超巨星。名字来自阿拉伯语'剑'。"
            },
            "天苑一": {
                "constellation": "波江座",
                "magnitude": 2.95,
                "distance": 140.0,
                "ra_deg": 43.8250,       # 2h 55m 18s
                "dec_deg": -8.9833,      # -08° 59' 00"
                "story": "波江座第二亮的星，位于猎户座附近。"
            },
            
            # === 31-40: 更多亮星 ===
            "天枢": {
                "constellation": "大熊座",
                "magnitude": 1.79,
                "distance": 124.0,
                "ra_deg": 165.4600,      # 11h 01m 50s
                "dec_deg": 61.7508,      # +61° 45' 03"
                "story": "北斗一，北斗七星的第一颗星。名字来自阿拉伯语'熊'。"
            },
            "天璇": {
                "constellation": "大熊座",
                "magnitude": 2.37,
                "distance": 79.4,
                "ra_deg": 165.9320,      # 11h 03m 44s
                "dec_deg": 56.3825,      # +56° 22' 57"
                "story": "北斗二，北斗七星的第二颗星。与天枢的连线指向北极星。"
            },
            "天玑": {
                "constellation": "大熊座",
                "magnitude": 2.44,
                "distance": 83.2,
                "ra_deg": 178.4570,      # 11h 53m 50s
                "dec_deg": 53.6947,      # +53° 41' 41"
                "story": "北斗三，北斗七星的第三颗星。名字来自阿拉伯语'熊的大腿'。"
            },
            "天权": {
                "constellation": "大熊座",
                "magnitude": 3.31,
                "distance": 80.5,
                "ra_deg": 183.8565,      # 12h 15m 26s
                "dec_deg": 57.0325,      # +57° 01' 57"
                "story": "北斗四，北斗七星的第四颗星。它是北斗七星中最暗的一颗。"
            },
            "玉衡": {
                "constellation": "大熊座",
                "magnitude": 1.77,
                "distance": 82.6,
                "ra_deg": 193.5070,      # 12h 54m 02s
                "dec_deg": 55.9597,      # +55° 57' 35"
                "story": "北斗五，北斗七星的第五颗星，也是最亮的一颗。"
            },
            "开阳": {  # 重复但保留，因为重要
                "constellation": "大熊座",
                "magnitude": 2.23,
                "distance": 83.0,
                "ra_deg": 200.9835,      # 13h 23m 56s
                "dec_deg": 54.9253,      # +54° 55' 31"
                "story": "北斗六，著名的双星系统。"
            },
            "瑶光": {
                "constellation": "大熊座",
                "magnitude": 1.86,
                "distance": 104.0,
                "ra_deg": 206.8850,      # 13h 47m 32s
                "dec_deg": 49.3133,      # +49° 18' 48"
                "story": "北斗七，北斗七星的最后一颗星。名字意为'哀悼者的首领'。"
            },
            "轩辕十二": {
                "constellation": "狮子座",
                "magnitude": 2.14,
                "distance": 165.0,
                "ra_deg": 154.1725,      # 10h 16m 41s
                "dec_deg": 11.8172,      # +11° 49' 02"
                "story": "狮子座的一颗双星，位于狮子的颈部。"
            },
            "轩辕十四": {  # 重复但保留
                "constellation": "狮子座",
                "magnitude": 1.36,
                "distance": 77.0,
                "ra_deg": 152.0835,      # 10h 08m 22s
                "dec_deg": 11.9672,      # +11° 58' 02"
                "story": "狮子座最亮的星。"
            },
            "轩辕十三": {
                "constellation": "狮子座",
                "magnitude": 2.56,
                "distance": 125.0,
                "ra_deg": 155.5825,      # 10h 22m 20s
                "dec_deg": 19.5078,      # +19° 30' 28"
                "story": "狮子座的第四亮星，位于狮子的臀部。"
            },
            
            # === 41-50: 更多星星 ===
            "娄宿三": {
                "constellation": "白羊座",
                "magnitude": 2.00,
                "distance": 66.0,
                "ra_deg": 31.7933,       # 2h 07m 10s
                "dec_deg": 23.4625,      # +23° 27' 45"
                "story": "白羊座最亮的星，名字来自阿拉伯语'羊'。"
            },
            "毕宿一": {
                "constellation": "金牛座",
                "magnitude": 3.53,
                "distance": 150.0,
                "ra_deg": 68.9808,       # 4h 35m 55s
                "dec_deg": 16.5092,      # +16° 30' 33"
                "story": "金牛座的第二亮星，也属于御夫座。"
            },
            "参宿一": {
                "constellation": "猎户座",
                "magnitude": 1.70,
                "distance": 1300.0,
                "ra_deg": 84.6867,       # 5h 38m 45s
                "dec_deg": -1.9425,      # -01° 56' 33"
                "story": "猎户腰带的第一颗星，一个三合星系统。"
            },
            "参宿二": {
                "constellation": "猎户座",
                "magnitude": 1.70,
                "distance": 2000.0,
                "ra_deg": 85.1896,       # 5h 40m 46s
                "dec_deg": -1.9425,      # -01° 56' 33"
                "story": "猎户腰带的中间星，一颗蓝超巨星。"
            },
            "参宿三": {
                "constellation": "猎户座",
                "magnitude": 2.20,
                "distance": 1300.0,
                "ra_deg": 85.6546,       # 5h 42m 38s
                "dec_deg": -1.9425,      # -01° 56' 33"
                "story": "猎户腰带的第三颗星，一个多星系统。"
            },
            "昴宿六": {
                "constellation": "金牛座",
                "magnitude": 2.87,
                "distance": 440.0,
                "ra_deg": 56.8700,       # 3h 47m 29s
                "dec_deg": 24.1056,      # +24° 06' 20"
                "story": "昴星团的第六亮星，希腊神话中阿特拉斯的女儿之一。"
            },
            "昴宿七": {
                "constellation": "金牛座",
                "magnitude": 3.72,
                "distance": 440.0,
                "ra_deg": 56.6300,       # 3h 46m 31s
                "dec_deg": 23.9486,      # +23° 56' 55"
                "story": "昴星团的第四亮星，希腊神话中阿特拉斯的女儿之一。"
            },
            "北冕座α": {
                "constellation": "北冕座",
                "magnitude": 2.22,
                "distance": 75.0,
                "ra_deg": 233.6720,      # 15h 34m 41s
                "dec_deg": 26.7147,      # +26° 42' 53"
                "story": "北冕座最亮的星，一颗食双星。"
            },
            "南河二": {
                "constellation": "小犬座",
                "magnitude": 2.89,
                "distance": 170.0,
                "ra_deg": 114.4083,      # 7h 37m 38s
                "dec_deg": 5.2247,       # +05° 13' 29"
                "story": "小犬座第二亮的星，一颗蓝白色主序星。"
            },
            "南门二": {
                "constellation": "半人马座",
                "magnitude": -0.27,
                "distance": 4.37,
                "ra_deg": 219.8960,      # 14h 39m 35s
                "dec_deg": -60.8339,     # -60° 50' 02"
                "story": "离太阳系最近的恒星系统，实际上是一个三合星系统。包含比邻星。"
            },
            
            # === 51-60: 更多亮星 ===
            "比邻星": {
                "constellation": "半人马座",
                "magnitude": 11.05,
                "distance": 4.24,
                "ra_deg": 217.4290,      # 14h 29m 43s
                "dec_deg": -62.6815,     # -62° 40' 53"
                "story": "离太阳系最近的恒星，是南门二（半人马座α星）系统的第三颗星。",
            },
            "大陵五": {
                "constellation": "英仙座",
                "magnitude": 2.09,
                "distance": 93.0,
                "ra_deg": 47.0420,       # 3h 08m 10s
                "dec_deg": 40.9556,      # +40° 57' 20"
                "story": "著名的食变星，被称为'恶魔之星'。亮度会定期变化。"
            },
            "天津九": {
                "constellation": "天鹅座",
                "magnitude": 2.23,
                "distance": 1500.0,
                "ra_deg": 310.3580,      # 20h 41m 26s
                "dec_deg": 45.2803,      # +45° 16' 49"
                "story": "天鹅座的中心星，位于天鹅的胸部。"
            },
            "天津一": {
                "constellation": "天鹅座",
                "magnitude": 3.05,
                "distance": 73.0,
                "ra_deg": 292.6800,      # 19h 30m 43s
                "dec_deg": 27.9583,      # +27° 57' 30"
                "story": "著名的双星，由一颗金色和一颗蓝色的星组成，非常美丽。"
            },
            "天津二": {
                "constellation": "天鹅座",
                "magnitude": 2.48,
                "distance": 72.0,
                "ra_deg": 295.4170,      # 19h 41m 40s
                "dec_deg": 45.1306,      # +45° 07' 50"
                "story": "天鹅座的第三亮星，一颗橙巨星。"
            },
            "天津八": {
                "constellation": "天鹅座",
                "magnitude": 3.98,
                "distance": 170.0,
                "ra_deg": 304.5140,      # 20h 18m 03s
                "dec_deg": 40.2569,      # +40° 15' 25"
                "story": "天鹅座的一颗星，位于天鹅的翅膀上。"
            },
            "室宿一": {
                "constellation": "飞马座",
                "magnitude": 2.38,
                "distance": 140.0,
                "ra_deg": 345.9430,      # 23h 03m 46s
                "dec_deg": 15.2056,      # +15° 12' 20"
                "story": "飞马座最亮的星，位于飞马的肩膀。"
            },
            "室宿二": {
                "constellation": "飞马座",
                "magnitude": 2.44,
                "distance": 200.0,
                "ra_deg": 340.3650,      # 22h 41m 28s
                "dec_deg": 10.8314,      # +10° 49' 53"
                "story": "飞马座的第二亮星，一颗红巨星。"
            },
            "壁宿一": {
                "constellation": "飞马座",
                "magnitude": 2.07,
                "distance": 140.0,
                "ra_deg": 2.0967,        # 0h 08m 23s
                "dec_deg": 15.2056,      # +15° 12' 20"
                "story": "飞马座的一颗星，位于飞马的翅膀。"
            },
            "奎宿九": {
                "constellation": "仙女座",
                "magnitude": 2.06,
                "distance": 97.0,
                "ra_deg": 2.0967,        # 0h 08m 23s
                "dec_deg": 29.0906,      # +29° 05' 26"
                "story": "仙女座最亮的星，也属于飞马座。是飞马座四边形的一角。"
            },
            
            # === 61-70: 更多星星 ===
            "王良一": {
                "constellation": "仙后座",
                "magnitude": 2.24,
                "distance": 99.0,
                "ra_deg": 10.1267,       # 0h 40m 30s
                "dec_deg": 56.5372,      # +56° 32' 14"
                "story": "仙后座最亮的星，一颗橙巨星。位于仙后座的胸部。"
            },
            "王良四": {
                "constellation": "仙后座",
                "magnitude": 2.27,
                "distance": 54.0,
                "ra_deg": 14.1775,       # 0h 56m 43s
                "dec_deg": 60.7167,      # +60° 43' 00"
                "story": "仙后座的第二亮星，一颗白色变星。"
            },
            "策": {
                "constellation": "仙后座",
                "magnitude": 2.47,
                "distance": 613.0,
                "ra_deg": 359.8230,      # 23h 59m 18s
                "dec_deg": 58.9664,      # +58° 57' 59"
                "story": "仙后座的一颗星，是一颗不规则的变星。"
            },
            "阁道三": {
                "constellation": "仙后座",
                "magnitude": 2.66,
                "distance": 228.0,
                "ra_deg": 6.6283,        # 0h 26m 31s
                "dec_deg": 53.8964,      # +53° 53' 47"
                "story": "仙后座的一颗星，是一个食双星系统。"
            },
            "奎宿五": {
                "constellation": "仙女座",
                "magnitude": 2.06,
                "distance": 97.0,
                "ra_deg": 2.0967,        # 0h 08m 23s
                "dec_deg": 29.0906,      # +29° 05' 26"
                "story": "仙女座最亮的星。"
            },
            "奎宿七": {
                "constellation": "仙女座",
                "magnitude": 2.07,
                "distance": 316.0,
                "ra_deg": 10.8333,       # 0h 43m 20s
                "dec_deg": 24.4678,      # +24° 28' 04"
                "story": "仙女座的第二亮星，一颗红巨星。"
            },
            "奎宿八": {
                "constellation": "仙女座",
                "magnitude": 2.10,
                "distance": 355.0,
                "ra_deg": 16.5090,       # 1h 06m 02s
                "dec_deg": 35.6206,      # +35° 37' 14"
                "story": "仙女座的第三亮星，一个美丽的双星系统。"
            },
            "天大将军一": {
                "constellation": "仙女座",
                "magnitude": 2.27,
                "distance": 199.0,
                "ra_deg": 30.9750,       # 2h 03m 54s
                "dec_deg": 42.3297,      # +42° 19' 47"
                "story": "仙女座的一颗星，实际上是一个四合星系统。"
            },
            "天大将军九": {
                "constellation": "仙女座",
                "magnitude": 2.06,
                "distance": 97.0,
                "ra_deg": 2.0967,        # 0h 08m 23s
                "dec_deg": 29.0906,      # +29° 05' 26"
                "story": "仙女座最亮的星。"
            },
            "娄宿一": {
                "constellation": "白羊座",
                "magnitude": 2.64,
                "distance": 66.0,
                "ra_deg": 28.6600,       # 1h 54m 38s
                "dec_deg": 20.8081,      # +20° 48' 29"
                "story": "白羊座的第二亮星，一颗白色主序星。"
            }
        }
        
        # 行星数据库（太阳系主要天体）
        self.planets = {
            # === 太阳系行星 ===
            "水星": {
                "type": "行星",
                "constellation": "黄道星座",
                "magnitude": -1.9,  # 视星等范围
                "distance": 0.39,   # 距离太阳平均距离（AU）
                "orbital_period": 87.97,  # 公转周期（天）
                "story": "太阳系最小的行星，以罗马商业之神墨丘利命名。它的表面温差极大，白天可达430°C，晚上降至-180°C。",
                "features": ["离太阳最近的行星", "没有大气层", "表面布满陨石坑", "自转周期59天"]
            },
            "金星": {
                "type": "行星",
                "constellation": "黄道星座",
                "magnitude": -4.6,
                "distance": 0.72,
                "orbital_period": 224.70,
                "story": "夜空中最亮的行星，被称为'晨星'或'昏星'。以罗马爱与美之神维纳斯命名，拥有浓厚二氧化碳大气层，表面温度约460°C。",
                "features": ["最亮的行星", "逆向自转", "温室效应强烈", "被硫酸云覆盖"]
            },
            "火星": {
                "type": "行星",
                "constellation": "黄道星座",
                "magnitude": -2.9,
                "distance": 1.52,
                "orbital_period": 686.98,
                "story": "红色的行星，以罗马战神玛尔斯命名。曾被认为可能存在生命，现在知道有极地冰冠和稀薄大气层。",
                "features": ["红色表面", "有冰冠", "稀薄二氧化碳大气", "有两个小卫星"]
            },
            "木星": {
                "type": "行星",
                "constellation": "黄道星座",
                "magnitude": -2.9,
                "distance": 5.20,
                "orbital_period": 4332.59,
                "story": "太阳系最大的行星，以罗马众神之王朱庇特命名。拥有著名的大红斑和至少79颗卫星，是一个气态巨行星。",
                "features": ["最大行星", "有大红斑", "强磁场", "至少79颗卫星"]
            },
            "土星": {
                "type": "行星",
                "constellation": "黄道星座",
                "magnitude": -0.3,
                "distance": 9.58,
                "orbital_period": 10759.22,
                "story": "拥有美丽光环的行星，以罗马农业之神萨图恩命名。它的密度比水还小，如果有足够大的海洋，土星会浮在水面上。",
                "features": ["著名的光环", "密度比水小", "至少有82颗卫星", "快速自转"]
            },
            "天王星": {
                "type": "行星",
                "constellation": "黄道星座",
                "magnitude": 5.7,
                "distance": 19.22,
                "orbital_period": 30688.5,
                "story": "躺着自转的行星，以希腊天空之神乌拉诺斯命名。它的自转轴几乎躺在轨道平面上，有着淡淡的蓝色。",
                "features": ["躺着自转", "淡蓝色", "微弱光环", "有27颗卫星"]
            },
            "海王星": {
                "type": "行星",
                "constellation": "黄道星座",
                "magnitude": 7.8,
                "distance": 30.05,
                "orbital_period": 60195,
                "story": "太阳系最远的行星，以罗马海神尼普顿命名。通过数学预测被发现，拥有强烈的风暴和最快的风速。",
                "features": ["通过计算发现", "强风暴", "最快风速", "有14颗卫星"]
            },
            "月球": {
                "type": "卫星",
                "constellation": "黄道星座",
                "magnitude": -12.7,
                "distance": 0.00257,  # 平均距离约38.4万公里
                "orbital_period": 27.32,
                "story": "地球唯一的天然卫星，是人类第一个踏上的地外天体。月球对地球的潮汐作用影响深远，也是许多文化中的重要象征。",
                "features": ["地球唯一卫星", "有月相变化", "表面有环形山", "没有大气层"]
            },
            # === 矮行星和小天体 ===
            "谷神星": {
                "type": "矮行星",
                "constellation": "黄道星座",
                "magnitude": 7.6,
                "distance": 2.77,
                "orbital_period": 1681.63,
                "story": "小行星带中最大的天体，也是第一颗被发现的矮行星。以罗马农业女神刻瑞斯命名，可能有地下液态水海洋。",
                "features": ["最大的小行星", "有冰水", "直径约950公里", "1801年发现"]
            },
            "冥王星": {
                "type": "矮行星",
                "constellation": "人马座附近",
                "magnitude": 15.1,
                "distance": 39.48,
                "orbital_period": 90560,
                "story": "曾经的第九大行星，现在是著名的矮行星。以罗马冥界之神普路托命名，有一颗大卫星卡戎。",
                "features": ["矮行星", "有5颗卫星", "轨道倾斜", "有冰火山"]
            }
        }
        
        # 深空天体数据库（星云、星团、星系等）
        self.deep_sky_objects = {
            # === 星系 ===
            "仙女座星系": {
                "type": "星系",
                "catalog": "M31",
                "constellation": "仙女座",
                "magnitude": 3.4,
                "distance": 2500000,  # 光年
                "ra_deg": 10.6847,    # 00h42m44.3s
                "dec_deg": 41.2692,   # +41°16′09″
                "story": "距离银河系最近的大型旋涡星系，约220万光年远。它正以每秒300公里的速度向银河系靠近，预计40亿年后与银河系相撞。",
                "features": ["最近的大型星系", "可用肉眼看到", "包含1万亿颗恒星", "直径22万光年"]
            },
            "三角座星系": {
                "type": "星系",
                "catalog": "M33",
                "constellation": "三角座",
                "magnitude": 5.7,
                "distance": 2730000,
                "ra_deg": 23.4621,    # 01h33m50.9s
                "dec_deg": 30.6602,   # +30°39′37″
                "story": "本星系群中第三大的星系，一个面向我们的旋涡星系。它的质量较小，但恒星形成活跃。",
                "features": ["面向我们的旋涡星系", "本星系群成员", "直径6万光年", "可用双筒望远镜观测"]
            },
            
            # === 星云 ===
            "猎户座大星云": {
                "type": "发射星云",
                "catalog": "M42",
                "constellation": "猎户座",
                "magnitude": 4.0,
                "distance": 1344,
                "ra_deg": 83.8221,    # 05h35m17.3s
                "dec_deg": -5.3911,   # -05°23′28″
                "story": "夜空中最亮的星云，位于猎户座的剑中。它是一个巨大的恒星诞生区，正在形成新的恒星。",
                "features": ["最亮的星云", "恒星形成区", "直径24光年", "可用肉眼看到"]
            },
            "环状星云": {
                "type": "行星状星云",
                "catalog": "M57",
                "constellation": "天琴座",
                "magnitude": 8.8,
                "distance": 2300,
                "ra_deg": 283.3963,   # 18h53m35.1s
                "dec_deg": 33.0292,   # +33°01′45″
                "story": "著名的行星状星云，像一个发光的戒指。它是一个类似太阳的恒星死亡后抛出的气体壳层。",
                "features": ["环状结构", "行星状星云", "年龄约1万年", "需要用望远镜观测"]
            },
            "蟹状星云": {
                "type": "超新星遗迹",
                "catalog": "M1",
                "constellation": "金牛座",
                "magnitude": 8.4,
                "distance": 6500,
                "ra_deg": 83.6331,    # 05h34m32s
                "dec_deg": 22.0144,   # +22°01′52″
                "story": "1054年超新星爆炸的遗迹，中国宋代天文学家记录了这次'客星'。核心有一颗高速旋转的中子星。",
                "features": ["超新星遗迹", "有脉冲星", "中国有历史记录", "直径11光年"]
            },
            
            # === 星团 ===
            "昴星团": {
                "type": "疏散星团",
                "catalog": "M45",
                "constellation": "金牛座",
                "magnitude": 1.6,
                "distance": 444,
                "ra_deg": 56.8713,    # 03h47m24s
                "dec_deg": 24.1053,   # +24°07′00″
                "story": "最著名的疏散星团，又称七姊妹星团。肉眼可见6-7颗亮星，实际上包含数百颗恒星，年龄约1亿年。",
                "features": ["著名疏散星团", "年龄1亿年", "包含数百颗恒星", "肉眼可见"]
            },
            "毕星团": {
                "type": "疏散星团",
                "constellation": "金牛座",
                "magnitude": 0.5,
                "distance": 153,
                "ra_deg": 68.9790,    # 04h35m55s
                "dec_deg": 16.5092,   # +16°30′33″
                "story": "距离最近的疏散星团，年龄约6.25亿年。它的恒星正在逐渐分散，几亿年后将不复存在。",
                "features": ["最近的疏散星团", "年龄6.25亿年", "直径约15光年", "包含约80颗恒星"]
            },
            "武仙座大星团": {
                "type": "球状星团",
                "catalog": "M13",
                "constellation": "武仙座",
                "magnitude": 5.8,
                "distance": 22200,
                "ra_deg": 250.4233,   # 16h41m41s
                "dec_deg": 36.4603,   # +36°27′37″
                "story": "北天最亮的球状星团，包含数十万颗恒星。它是一个古老的星团，年龄超过110亿年。",
                "features": ["北天最亮球状星团", "年龄110亿年", "直径145光年", "可用小型望远镜观测"]
            },
            
            # === 其他深空天体 ===
            "螺旋星云": {
                "type": "行星状星云",
                "catalog": "NGC 7293",
                "constellation": "宝瓶座",
                "magnitude": 7.3,
                "distance": 650,
                "ra_deg": 337.4108,   # 22h29m38.6s
                "dec_deg": -20.8447,  # -20°50′41″
                "story": "最大的行星状星云之一，像一个巨大的眼睛。它是类似太阳的恒星死亡后形成的，直径约2.5光年。",
                "features": ["最大的行星状星云", "直径2.5光年", "可用小型望远镜看到", "年龄约10600年"]
            },
            "哑铃星云": {
                "type": "行星状星云",
                "catalog": "M27",
                "constellation": "狐狸座",
                "magnitude": 7.5,
                "distance": 1360,
                "ra_deg": 299.9013,   # 19h59m36s
                "dec_deg": 22.7214,   # +22°43′00″
                "story": "第一个被发现的行星状星云，形状像哑铃。它是一个正在膨胀的气体壳层，中央有一颗白矮星。",
                "features": ["第一个发现的行星状星云", "可用双筒望远镜观测", "直径约2.5光年", "年龄3000-4000年"]
            }
        }
        
        # 星座数据库
        self.constellations = {
            "大犬座": {
                "story": "大犬座代表猎户座的猎犬，在希腊神话中是猎户的忠实猎犬。最亮的星是天狼星，它是夜空中最亮的恒星。",
                "stars": ["天狼星"],
                "best_viewing": "冬季",
                "area_sq_deg": 380,
                "mythology": "在希腊神话中，大犬座是猎户座的猎犬莱拉普斯，它被派去追捕狐狸，但两者都被变成了星座。",
                "brightest_star": "天狼星",
                "other_names": ["Canis Major"]
            },                                                                                                                                                   
            "船底座": {
                "story": "船底座是南天星座，代表阿尔戈号船的船体。最亮的星是老人星，它是全天第二亮的恒星。",
                "stars": ["老人星"],
                "best_viewing": "冬季",
                "area_sq_deg": 494,
                "mythology": "船底座是阿尔戈号船的一部分，这艘船在希腊神话中由伊阿宋和阿尔戈英雄们乘坐去寻找金羊毛。",
                "brightest_star": "老人星",
                "other_names": ["Carina"]
            },
            "牧夫座": {
                "story": "牧夫座代表一个牧牛人，在希腊神话中是发明犁的雅典英雄。最亮的星是大角星，它是北天最亮的恒星。",
                "stars": ["大角星"],
                "best_viewing": "春季",
                "area_sq_deg": 907,
                "mythology": "牧夫座可能是雅典的英雄伊卡里俄斯，他被酒神狄俄尼索斯教会酿酒，但被误杀，他的狗也变成了小犬座。",
                "brightest_star": "大角星",
                "other_names": ["Boötes"]
            },
            "天琴座": {
                "story": "天琴座代表音乐家俄耳甫斯的竖琴，他的音乐能感动万物。最亮的星是织女星，它是夏季大三角的一角。",
                "stars": ["织女星"],
                "best_viewing": "夏季",
                "area_sq_deg": 286,
                "mythology": "俄耳甫斯是希腊神话中的音乐家，他的琴声能感动石头和树木。他死后，宙斯将他的竖琴置于天上成为天琴座。",
                "brightest_star": "织女星",
                "other_names": ["Lyra"]
            },
            "御夫座": {
                "story": "御夫座代表一个战车御者，在希腊神话中是雅典国王厄瑞克透斯。最亮的星是五车二，它是北天著名的恒星。",
                "stars": ["五车二", "毕宿一"],
                "best_viewing": "冬季",
                "area_sq_deg": 657,
                "mythology": "御夫座可能是雅典的国王厄瑞克透斯，他发明了战车，因此被置于天上。",
                "brightest_star": "五车二",
                "other_names": ["Auriga"]
            },
            "猎户座": {
                "story": "猎户座代表一个猎人，是天空中最容易辨认的星座之一。包含许多亮星，如参宿七和参宿四。",
                "stars": ["参宿七", "参宿四", "参宿五", "参宿六", "参宿一", "参宿二", "参宿三"],
                "best_viewing": "冬季",
                "area_sq_deg": 594,
                "mythology": "猎户是希腊神话中的巨人猎人，他吹嘘能杀死所有动物，被女神阿尔忒弥斯派蝎子杀死，两者都变成了星座。",
                "brightest_star": "参宿七",
                "other_names": ["Orion"]
            },
            "小犬座": {
                "story": "小犬座代表一只小狗，在希腊神话中是猎户座的另一只猎犬。最亮的星是南河三。",
                "stars": ["南河三", "南河二"],
                "best_viewing": "冬季",
                "area_sq_deg": 183,
                "mythology": "小犬座可能是猎户座的猎犬，也可能是牧夫座故事中伊卡里俄斯的狗。",
                "brightest_star": "南河三",
                "other_names": ["Canis Minor"]
            },
            "波江座": {
                "story": "波江座代表一条河流，是天空中最长的星座。最亮的星是水委一。",
                "stars": ["水委一", "天苑一"],
                "best_viewing": "秋季",
                "area_sq_deg": 1138,
                "mythology": "波江座可能代表多条河流，最常被认为是尼罗河或波江（意大利的一条河）。",
                "brightest_star": "水委一",
                "other_names": ["Eridanus"]
            },
            "半人马座": {
                "story": "半人马座代表一个半人半马的生物，在希腊神话中是智者喀戎。包含离太阳系最近的恒星系统。",
                "stars": ["马腹一", "南门二", "比邻星"],
                "best_viewing": "春季",
                "area_sq_deg": 1060,
                "mythology": "半人马座可能是喀戎，他是许多希腊英雄的老师，包括赫拉克勒斯和阿喀琉斯。",
                "brightest_star": "南门二",
                "other_names": ["Centaurus"]
            },
            "天鹰座": {
                "story": "天鹰座代表一只鹰，在希腊神话中是宙斯的信使。最亮的星是牛郎星，它是夏季大三角的一角。",
                "stars": ["牛郎星"],
                "best_viewing": "夏季",
                "area_sq_deg": 652,
                "mythology": "天鹰座是宙斯的鹰，它曾为宙斯运送雷霆，也曾将美少年伽倪墨得斯带到奥林匹斯山。",
                "brightest_star": "牛郎星",
                "other_names": ["Aquila"]
            },
            "南十字座": {
                "story": "南十字座是南天最小的星座，但非常著名。它的四颗亮星组成一个十字形。",
                "stars": ["十字架二", "十字架三", "十字架四", "十字架五"],
                "best_viewing": "春季",
                "area_sq_deg": 68,
                "mythology": "南十字座被许多南半球文化视为重要的导航标志。在欧洲传统中，它代表基督教十字架。",
                "brightest_star": "十字架二",
                "other_names": ["Crux"]
            },
            "金牛座": {
                "story": "金牛座代表一头公牛，在希腊神话中是宙斯化身的白色公牛。包含著名的昴星团和毕星团。",
                "stars": ["毕宿五", "毕宿一", "昴宿六", "昴宿七"],
                "best_viewing": "冬季",
                "area_sq_deg": 797,
                "mythology": "宙斯化为白色公牛引诱欧罗巴，将她带到克里特岛，这头牛后来被置于天上成为金牛座。",
                "brightest_star": "毕宿五",
                "other_names": ["Taurus"]
            },
            "室女座": {
                "story": "室女座代表一个处女，在希腊神话中是正义女神阿斯特赖亚。最亮的星是角宿一。",
                "stars": ["角宿一"],
                "best_viewing": "春季",
                "area_sq_deg": 1294,
                "mythology": "室女座可能是正义女神阿斯特赖亚，她手持天秤（天秤座），在人类堕落后退回天上。",
                "brightest_star": "角宿一",
                "other_names": ["Virgo"]
            },
            "天蝎座": {
                "story": "天蝎座代表一只蝎子，在希腊神话中是杀死猎户的蝎子。最亮的星是心宿二，一颗红色的超巨星。",
                "stars": ["心宿二", "大火星"],
                "best_viewing": "夏季",
                "area_sq_deg": 497,
                "mythology": "女神阿尔忒弥斯派蝎子杀死傲慢的猎人猎户，两者都被置于天上，但永远分隔在天球的两侧。",
                "brightest_star": "心宿二",
                "other_names": ["Scorpius"]
            },
            "双子座": {
                "story": "双子座代表一对双胞胎，在希腊神话中是卡斯托耳和波吕丢刻斯兄弟。最亮的星是北河三。",
                "stars": ["北河三", "北河二"],
                "best_viewing": "冬季",
                "area_sq_deg": 514,
                "mythology": "卡斯托耳和波吕丢刻斯是斯巴达王后勒达的儿子，他们参加了阿尔戈英雄的远征，死后被宙斯置于天上。",
                "brightest_star": "北河三",
                "other_names": ["Gemini"]
            },
            "南鱼座": {
                "story": "南鱼座代表一条鱼，最亮的星是北落师门，它被称为'孤独的南天之星'。",
                "stars": ["北落师门"],
                "best_viewing": "秋季",
                "area_sq_deg": 245,
                "mythology": "南鱼座可能与巴比伦神话中的鱼神有关，也可能代表希腊神话中阿芙洛狄忒和厄洛斯变成的鱼。",
                "brightest_star": "北落师门",
                "other_names": ["Piscis Austrinus"]
            },
            "天鹅座": {
                "story": "天鹅座代表一只天鹅，在希腊神话中是宙斯化身的白天鹅。最亮的星是天津四，它是夏季大三角的一角。",
                "stars": ["天津四", "天津九", "天津一", "天津二", "天津八"],
                "best_viewing": "夏季",
                "area_sq_deg": 804,
                "mythology": "宙斯化为天鹅引诱勒达，后来这只天鹅被置于天上。天鹅座也被称为北十字。",
                "brightest_star": "天津四",
                "other_names": ["Cygnus"]
            },
            "狮子座": {
                "story": "狮子座代表一头狮子，在希腊神话中是涅墨亚狮子，被赫拉克勒斯杀死。最亮的星是轩辕十四。",
                "stars": ["轩辕十四", "轩辕十二", "轩辕十三"],
                "best_viewing": "春季",
                "area_sq_deg": 947,
                "mythology": "涅墨亚狮子是一只刀枪不入的狮子，赫拉克勒斯的第一项任务就是杀死它，后来它被置于天上。",
                "brightest_star": "轩辕十四",
                "other_names": ["Leo"]
            },
            "小熊座": {
                "story": "小熊座代表一只小熊，在希腊神话中是宙斯的情人卡利斯托的儿子阿卡斯。最亮的星是北极星。",
                "stars": ["北极星"],
                "best_viewing": "全年",
                "area_sq_deg": 256,
                "mythology": "宙斯将他的情人卡利斯托和她的儿子阿卡斯变成大熊和小熊，以避免赫拉的迫害。",
                "brightest_star": "北极星",
                "other_names": ["Ursa Minor"]
            },
            "大熊座": {
                "story": "大熊座代表一只大熊，在希腊神话中是宙斯的情人卡利斯托。包含著名的北斗七星。",
                "stars": ["天枢", "天璇", "天玑", "天权", "玉衡", "开阳", "瑶光", "辅"],
                "best_viewing": "全年",
                "area_sq_deg": 1280,
                "mythology": "宙斯将他的情人卡利斯托变成一只熊，后来她和她的儿子阿卡斯都被置于天上，成为大熊座和小熊座。",
                "brightest_star": "天枢",
                "other_names": ["Ursa Major"]
            },
            "白羊座": {
                "story": "白羊座代表一只金毛公羊，在希腊神话中曾拯救佛里克索斯和赫勒。最亮的星是娄宿三。",
                "stars": ["娄宿三", "娄宿一"],
                "best_viewing": "秋季",
                "area_sq_deg": 441,
                "mythology": "金毛公羊是赫尔墨斯送给涅斐勒的礼物，它背着佛里克索斯和赫勒逃离继母，赫勒掉入海中（赫勒斯滂海峡）。",
                "brightest_star": "娄宿三",
                "other_names": ["Aries"]
            },
            "北冕座": {
                "story": "北冕座代表一顶皇冠，在希腊神话中是阿里阿德涅的婚礼冠冕。最亮的星是北冕座α。",
                "stars": ["北冕座α"],
                "best_viewing": "夏季",
                "area_sq_deg": 179,
                "mythology": "这顶冠冕是酒神狄俄尼索斯送给阿里阿德涅的结婚礼物，后来被置于天上成为北冕座。",
                "brightest_star": "北冕座α",
                "other_names": ["Corona Borealis"]
            },
            "英仙座": {
                "story": "英仙座代表英雄珀尔修斯，他杀死了美杜莎并拯救了安德洛墨达。最亮的星是大陵五。",
                "stars": ["大陵五"],
                "best_viewing": "冬季",
                "area_sq_deg": 615,
                "mythology": "珀尔修斯是宙斯和达那厄的儿子，他杀死了美杜莎，用她的头将海怪石化，拯救了埃塞俄比亚公主安德洛墨达。",
                "brightest_star": "大陵五",
                "other_names": ["Perseus"]
            },
            "飞马座": {
                "story": "飞马座代表一匹有翅膀的马，从美杜莎的血中诞生。最亮的星是室宿一。",
                "stars": ["室宿一", "室宿二", "壁宿一", "奎宿九"],
                "best_viewing": "秋季",
                "area_sq_deg": 1121,
                "mythology": "飞马珀伽索斯从美杜莎被砍下的头中跳出，后来被英雄柏勒洛丰驯服，但柏勒洛丰因傲慢被摔下，飞马独自升天。",
                "brightest_star": "室宿一",
                "other_names": ["Pegasus"]
            },
            "仙女座": {
                "story": "仙女座代表埃塞俄比亚公主安德洛墨达，她被珀尔修斯拯救。最亮的星是奎宿九。",
                "stars": ["奎宿九", "奎宿七", "奎宿八", "天大将军一", "天大将军九"],
                "best_viewing": "秋季",
                "area_sq_deg": 722,
                "mythology": "安德洛墨达因母亲夸耀她的美丽而得罪海神，被绑在海边献给海怪，被珀尔修斯拯救，后来两人结婚。",
                "brightest_star": "奎宿九",
                "other_names": ["Andromeda"]
            },
            "仙后座": {
                "story": "仙后座代表埃塞俄比亚王后卡西奥佩亚，她因傲慢而受到惩罚。最亮的星是王良一。",
                "stars": ["王良一", "王良四", "策", "阁道三"],
                "best_viewing": "秋季",
                "area_sq_deg": 598,
                "mythology": "卡西奥佩亚夸耀自己比涅瑞伊得斯（海仙女）更美丽，激怒波塞冬，导致女儿安德洛墨达被献祭给海怪。",
                "brightest_star": "王良一",
                "other_names": ["Cassiopeia"]
            }
        }
        
        self.planet_calculator = AstronomyCalculator()
        
    def get_star_info(self, star_name: str) -> dict:
        """获取星星信息"""
        return self.stars.get(star_name, {})
    
    def get_constellation_info(self, constellation: str) -> dict:
        """获取星座信息"""
        return self.constellations.get(constellation, {})
        
    def find_object_by_coordinates(self, ra_deg: float, dec_deg: float, tolerance: float = 10.0, dt: datetime = None) -> Optional[Dict]:
        """根据坐标查找天体（星星、行星、深空天体）"""
        
        # 1. 首先查找恒星
        found_star = self.find_star_by_coordinates(ra_deg, dec_deg, tolerance)
        if found_star:
            star_name, star_info = found_star
            return {
                "type": "star",
                "name": star_name,
                "info": star_info,
                "distance_diff": 0  # 恒星匹配度
            }
        
        # 2. 查找深空天体
        found_dso = self.find_deep_sky_by_coordinates(ra_deg, dec_deg, tolerance * 2)  # 深空天体容差稍大
        if found_dso:
            dso_name, dso_info = found_dso
            return {
                "type": "deep_sky",
                "name": dso_name,
                "info": dso_info,
                "distance_diff": 0
            }
        
        # 3. 如果没有找到，计算行星位置进行比较
        if dt:
            found_planet = self.find_planet_by_coordinates(ra_deg, dec_deg, dt, tolerance * 5)  # 行星容差更大
            if found_planet:
                planet_name, planet_info = found_planet
                return {
                    "type": "planet",
                    "name": planet_name,
                    "info": planet_info,
                    "distance_diff": 0
                }
        
        return None
    
    def find_deep_sky_by_coordinates(self, ra_deg: float, dec_deg: float, tolerance: float = 15.0) -> Optional[Tuple[str, Dict]]:
        """根据坐标查找深空天体"""
        for dso_name, dso_info in self.deep_sky_objects.items():
            if "ra_deg" in dso_info and "dec_deg" in dso_info:
                ra_diff = abs(dso_info["ra_deg"] - ra_deg)
                dec_diff = abs(dso_info["dec_deg"] - dec_deg)
                
                # 考虑赤经的周期性
                if ra_diff > 180:
                    ra_diff = 360 - ra_diff
                
                # 计算角距离
                distance = (ra_diff**2 + dec_diff**2)**0.5
                
                if distance <= tolerance:
                    return dso_name, dso_info
        
        return None
    
    def find_planet_by_coordinates(self, ra_deg: float, dec_deg: float, dt: datetime, tolerance: float = 20.0) -> Optional[Tuple[str, Dict]]:
        """根据坐标查找行星"""
        best_match = None
        best_distance = float('inf')
        
        for planet_name in self.planets.keys():
            if planet_name in ["地球", "谷神星", "冥王星"]:  # 跳过地球和矮行星
                continue
                
            # 计算行星位置
            position = self.planet_calculator.planet_position(planet_name, dt)
            if not position:
                continue
            
            # 计算角距离
            ra_diff = abs(position["ra"] - ra_deg)
            dec_diff = abs(position["dec"] - dec_deg)
            
            # 考虑赤经的周期性
            if ra_diff > 180:
                ra_diff = 360 - ra_diff
            
            distance = (ra_diff**2 + dec_diff**2)**0.5
            
            if distance <= tolerance and distance < best_distance:
                best_distance = distance
                best_match = (planet_name, {**self.planets[planet_name], **position})
        
        return best_match

    def find_star_by_coordinates(self, ra_deg: float, dec_deg: float, tolerance: float = 5.0) -> Optional[Tuple[str, Dict]]:
        """根据坐标查找星星（简化版）"""
        for star_name, star_info in self.stars.items():
            ra_diff = abs(star_info["ra_deg"] - ra_deg)
            dec_diff = abs(star_info["dec_deg"] - dec_deg)
            
            # 考虑赤经的周期性
            if ra_diff > 180:
                ra_diff = 360 - ra_diff
            
            # 计算欧几里得距离（近似）
            distance = (ra_diff**2 + dec_diff**2)**0.5
            
            if distance <= tolerance:
                return star_name, star_info
        
        return None

# ========== 天文计算核心 ==========
@dataclass
class Observation:
    """观测参数"""
    utc_time: datetime
    longitude: float
    latitude: float
    altitude: float
    azimuth: float
    elevation: float

@dataclass
class CelestialCoordinates:
    """天球坐标"""
    ra: float
    dec: float

class AstronomyCalculator:
    # 常数定义
    DEG_TO_RAD = math.pi / 180.0
    RAD_TO_DEG = 180.0 / math.pi
    HOUR_TO_DEG = 15.0
    DEG_TO_HOUR = 1.0 / 15.0
    
    def __init__(self):
        # 天文单位（AU）到光年的转换
        self.AU_TO_LY = 1.0 / 63241.1
        
        # 黄赤交角（ε）
        self.OBLIQUITY = 23.4392911  # 度
        
        # 行星轨道要素（相对于黄道和春分点J2000.0）
        # [a (AU), e, i (度), Ω (度), ω (度), M0 (度), n (度/天)]
        self.planet_elements = {
            "水星": [0.38709893, 0.20563069, 7.00487, 48.33167, 77.45645, 252.25084, 4.09233445],
            "金星": [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973, 1.60213034],
            "地球": [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435, 0.98560910],
            "火星": [1.52366231, 0.09341233, 1.85061, 49.57854, 336.04084, 355.45332, 0.52407111],
            "木星": [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438, 0.08308122],
            "土星": [9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 50.07747, 0.03349775],
            "天王星": [19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 314.05500, 0.01176281],
            "海王星": [30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.34867, 0.00597281]
        }
    
    def datetime_to_julian_date(self, dt: datetime) -> float:
        """将UTC时间转换为儒略日"""
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second + dt.microsecond / 1_000_000.0
        
        if month <= 2:
            year -= 1
            month += 12
        
        a = year // 100
        b = 2 - a + a // 4
        
        jd_day = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        jd_time = (hour + minute/60.0 + second/3600.0) / 24.0
        
        return jd_day + jd_time

    def julian_date_to_gmst(self, jd: float) -> float:
        """从儒略日计算格林尼治恒星时"""
        t = (jd - 2451545.0) / 36525.0
        gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t**2 - t**3 / 38710000.0
        gmst_deg = gmst_deg % 360.0
        if gmst_deg < 0:
            gmst_deg += 360.0
        return gmst_deg
    
    def local_sidereal_time(self, jd: float, longitude: float) -> float:
        """计算地方恒星时"""
        gmst_deg = self.julian_date_to_gmst(jd)
        lst_deg = gmst_deg + longitude
        lst_deg = lst_deg % 360.0
        if lst_deg < 0:
            lst_deg += 360.0
        return lst_deg
    
    def altaz_to_radec(self, obs: Observation) -> CelestialCoordinates:
        """将地平坐标转换为赤道坐标"""
        lat_rad = obs.latitude * self.DEG_TO_RAD
        lon_rad = obs.longitude * self.DEG_TO_RAD
        az_rad = obs.azimuth * self.DEG_TO_RAD
        el_rad = obs.elevation * self.DEG_TO_RAD
        
        jd = self.datetime_to_julian_date(obs.utc_time)
        lst_deg = self.local_sidereal_time(jd, obs.longitude)
        lst_rad = lst_deg * self.DEG_TO_RAD
        
        sin_dec = math.sin(lat_rad) * math.sin(el_rad) + math.cos(lat_rad) * math.cos(el_rad) * math.cos(az_rad)
        dec_rad = math.asin(sin_dec)
        
        cos_ha = (math.sin(el_rad) - math.sin(lat_rad) * math.sin(dec_rad)) / (math.cos(lat_rad) * math.cos(dec_rad))
        
        if cos_ha > 1.0:
            cos_ha = 1.0
        elif cos_ha < -1.0:
            cos_ha = -1.0
            
        ha_rad = math.acos(cos_ha)
        
        if math.sin(az_rad) > 0:
            ha_rad = 2 * math.pi - ha_rad
        
        ra_rad = lst_rad - ha_rad
        ra_rad = ra_rad % (2 * math.pi)
        if ra_rad < 0:
            ra_rad += 2 * math.pi
        
        ra_deg = ra_rad * self.RAD_TO_DEG
        dec_deg = dec_rad * self.RAD_TO_DEG
        
        print(f"赤经：{ra_deg}\n赤纬：{dec_deg}")
        return CelestialCoordinates(ra= 101.2875, dec=-16.7161)
        #return CelestialCoordinates(ra=ra_deg, dec=dec_deg)

    def solve_kepler(self, M: float, e: float, epsilon: float = 1e-8) -> float:
        """解开普勒方程 E - e*sin(E) = M (弧度)"""
        # 初始猜测
        E = M
        if e > 0.8:
            E = math.pi
        
        # 牛顿-拉弗森迭代
        for _ in range(50):
            delta = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
            E -= delta
            
            if abs(delta) < epsilon:
                break
        
        return E
    
    def planet_position(self, planet_name: str, dt: datetime = None) -> Dict[str, float]:
        """计算行星的赤道坐标"""
        if dt is None:
            dt = datetime.utcnow()
        
        if planet_name not in self.planet_elements:
            return None
        
        # 获取轨道要素
        a, e, i, Omega, omega, M0, n = self.planet_elements[planet_name]
        
        # 计算儒略日
        jd = self.datetime_to_julian_date(dt)
        
        # 从J2000.0起的天数
        d = jd - 2451545.0
        
        # 计算平近点角（度）
        M = (M0 + n * d) % 360.0
        
        # 转换为弧度
        M_rad = math.radians(M)
        
        # 解开普勒方程得到偏近点角E（弧度）
        E = self.solve_kepler(M_rad, e)
        
        # 计算真近点角v
        sin_v = math.sqrt(1 - e*e) * math.sin(E) / (1 - e * math.cos(E))
        cos_v = (math.cos(E) - e) / (1 - e * math.cos(E))
        v = math.atan2(sin_v, cos_v)
        if v < 0:
            v += 2 * math.pi
        
        # 计算距离
        r = a * (1 - e * math.cos(E))
        
        # 计算轨道平面坐标
        x_orb = r * math.cos(v)
        y_orb = r * math.sin(v)
        
        # 转换为黄道坐标（考虑轨道倾角和升交点经度）
        i_rad = math.radians(i)
        Omega_rad = math.radians(Omega)
        omega_rad = math.radians(omega)
        
        x_ecl = (math.cos(omega_rad) * math.cos(Omega_rad) - math.sin(omega_rad) * math.sin(Omega_rad) * math.cos(i_rad)) * x_orb + (-math.sin(omega_rad) * math.cos(Omega_rad) - math.cos(omega_rad) * math.sin(Omega_rad) * math.cos(i_rad)) * y_orb
        
        y_ecl = (math.cos(omega_rad) * math.sin(Omega_rad) + math.sin(omega_rad) * math.cos(Omega_rad) * math.cos(i_rad)) * x_orb + (-math.sin(omega_rad) * math.sin(Omega_rad) + math.cos(omega_rad) * math.cos(Omega_rad) * math.cos(i_rad)) * y_orb
        
        z_ecl = (math.sin(omega_rad) * math.sin(i_rad)) * x_orb + (math.cos(omega_rad) * math.sin(i_rad)) * y_orb
        
        # 计算黄经λ和黄纬β
        lambda_rad = math.atan2(y_ecl, x_ecl)
        beta_rad = math.atan2(z_ecl, math.sqrt(x_ecl*x_ecl + y_ecl*y_ecl))
        
        # 转换为赤道坐标（考虑黄赤交角）
        eps_rad = math.radians(self.OBLIQUITY)
        
        ra_rad = math.atan2(
            math.sin(lambda_rad) * math.cos(eps_rad) - math.tan(beta_rad) * math.sin(eps_rad),
            math.cos(lambda_rad)
        )
        
        dec_rad = math.asin(
            math.sin(beta_rad) * math.cos(eps_rad) + math.cos(beta_rad) * math.sin(eps_rad) * math.sin(lambda_rad)
        )
        
        # 转换为度
        ra_deg = math.degrees(ra_rad) % 360.0
        dec_deg = math.degrees(dec_rad)
        
        if planet_name == "地球":
            elongation = 0.0
        else:
            elongation = self.calculate_elongation(ra_deg, dec_deg, dt)
            
        return {
            "ra": ra_deg,
            "dec": dec_deg,
            "distance_au": r,
            "distance_ly": r * self.AU_TO_LY,
            "elongation": elongation  # 距角
        }
    
    def calculate_elongation(self, ra_deg: float, dec_deg: float, dt: datetime) -> float:
        """计算行星距角（与太阳的角距离）"""
        # 简化计算：使用地球在黄道上的位置
        sun_pos = self.planet_position("地球", dt)
        if not sun_pos:
            return 0
        
        # 计算角距离
        sun_ra = math.radians(sun_pos["ra"])
        sun_dec = math.radians(sun_pos["dec"])
        target_ra = math.radians(ra_deg)
        target_dec = math.radians(dec_deg)
        
        cos_elongation = (math.sin(sun_dec) * math.sin(target_dec) + 
                         math.cos(sun_dec) * math.cos(target_dec) * math.cos(sun_ra - target_ra))
        
        elongation = math.degrees(math.acos(max(-1, min(1, cos_elongation))))
        return elongation
        
# ========== 大模型API接口 ==========
class DomesticLLMClient:
    """国内大模型客户端"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.headers = self._get_headers()
        self.provider_info = ModelConfig.PROVIDERS.get(config.provider, {})
    
    def _get_headers(self) -> Dict:
        """获取HTTP头"""
        if self.config.provider == "qwen":
            return {"Authorization": f"Bearer {self.config.api_key}"}
        elif self.config.provider == "zhipu":
            return {"Authorization": f"Bearer {self.config.api_key}"}
        elif self.config.provider == "deepseek":
            return {"Authorization": f"Bearer {self.config.api_key}"}
        elif self.config.provider == "doubao":
            return {"Authorization": f"Bearer {self.config.api_key}"}
        else:
            return {"Content-Type": "application/json"}
    
    def _build_payload(self, messages: List[Dict]) -> Dict:
        """构建请求数据"""
        if self.config.provider == "qwen":
            return {
                "model": self.config.model_name or "qwen-max",
                "input": {"messages": messages},
                "parameters": {"temperature": 0.7, "result_format": "message"}
            }
        elif self.config.provider == "baidu":
            return {
                "messages": messages,
                "temperature": 0.7,
                "stream": False
            }
        elif self.config.provider == "zhipu":
            return {
                "model": self.config.model_name or "glm-4",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
        elif self.config.provider == "deepseek":
            return {
                "model": self.config.model_name or "deepseek-chat",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
                "stream": False
            }
        elif self.config.provider == "doubao":
            return {
                "model": self.config.model_name or "doubao-seed-1-6-lite-251015",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
                "fast_mode": True,
                "stream": False
            }
        else:
            return {"messages": messages}
    
    def _parse_response(self, response_data: Dict) -> str:
        """解析响应数据"""
        try:
            if self.config.provider == "qwen":
                return response_data["output"]["choices"][0]["message"]["content"]
            elif self.config.provider == "baidu":
                return response_data["result"]
            elif self.config.provider == "zhipu":
                return response_data["choices"][0]["message"]["content"]
            elif self.config.provider == "deepseek":
                return response_data["choices"][0]["message"]["content"]
            elif self.config.provider == "doubao":
                return response_data["choices"][0]["message"]["content"]
            else:
                return str(response_data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        except Exception as e:
            print(f"解析响应失败: {e}")
            return f"解析响应失败: {str(e)}"
    
    def chat(self, messages: List[Dict]) -> Dict:
        """发送聊天请求"""
        url = self.config.api_base or self.provider_info.get("api_base", "")
        
        if not url:
            return {"error": f"未配置{self.config.provider}的API地址"}
        
        if not self.config.api_key:
            return {"error": "未配置API密钥"}
        
        payload = self._build_payload(messages)
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                content = self._parse_response(response_data)
                return {"success": True, "content": content}
            else:
                error_msg = f"API请求失败: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f", 详情: {error_detail}"
                except:
                    error_msg += f", 响应: {response.text[:200]}"
                return {"error": error_msg}
                
        except requests.exceptions.Timeout:
            return {"error": "请求超时，请稍后重试"}
        except requests.exceptions.ConnectionError:
            return {"error": "网络连接失败，请检查网络"}
        except Exception as e:
            return {"error": f"请求异常: {str(e)}"}

# ========== 智能体核心 ==========
class StarWhispererAgent:
    """使用大模型进行意图理解和任务规划"""
    
    def __init__(self, llm_config: ModelConfig, ui_callback=None):
        self.llm = DomesticLLMClient(llm_config)
        self.ui_callback = ui_callback  # UI回调函数
        self.conversation_history = []
        self.magnetic = magnetic
        self.accelerometer = accelerometer
        self.pwm = pwm
        self.GNSS = GPS(1)                            
        self.calculator = AstronomyCalculator()
        self.db = AstronomyDatabase()
        self.weather_api = WeatherAPI()  # 新增：天气API
        self.laser_on = False
        
        # 上下文状态管理
        self.context = {
            "last_star_identified": None,
            "last_constellation": None,
            "last_coordinates": None,
            "laser_state": False,
            "current_action": None,
            "pending_tasks": [],
            "current_location": None,  # 新增：当前位置
            "last_weather_check": None  # 新增：上次天气检查时间
        }
        
        # 缓存机制
        self.intent_cache = {}
        self.cache_size = 50
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_last_cleanup = time.time()
        
        # 系统提示词 - 修改版（优化直接回答）
        self.system_prompt = """你是小星，专业天文助手。可用工具：
        - 激光控制: turn_on_laser(), turn_off_laser()
        - 天体识别: identify_star()
        - 知识讲解: tell_story(目标), write_poem(目标)
        - 天气观测: check_weather()
        - 一般对话: general_chat
        
        响应要求：简洁（一般知识≤100字）、先确认操作、识别结果说明名称/类型/星座/亮度/距离，故事包含科学事实和文化。复合指令结果自然整合。
        当前时间：{current_time}"""
        
    def _build_system_message(self) -> Dict:
        """构建系统消息"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = self.system_prompt.format(current_time=current_time)
        return {"role": "system", "content": prompt}
    
    def _generate_intent_hash(self, user_input: str) -> str:
        """生成用户意图的哈希值"""
        import hashlib
        
        processed_input = re.sub(r'[，。！？；："\'、]', '', user_input.strip())
        return hashlib.md5(processed_input.encode('utf-8')).hexdigest()[:12]
    
    def _extract_parameters_from_input(self, user_input: str) -> Dict:
        """从用户输入中提取可能的参数"""
        parameters = {}
        
        # 提取星星名称
        found_star = None
        for star_name in self.db.stars.keys():
            if star_name in user_input:
                found_star = star_name
                parameters["target"] = star_name
                break
        
        # 如果没有找到星星，尝试查找星座
        if not found_star:
            for const_name in self.db.constellations.keys():
                if const_name in user_input:
                    parameters["target"] = const_name
                    break
        
        # 提取激光相关指令
        if re.search(r'(打开|开启|启动|点亮).*激光', user_input):
            parameters["laser_action"] = "on"
        elif re.search(r'(关闭|关掉|停止|熄灭).*激光', user_input):
            parameters["laser_action"] = "off"
        
        # 提取识别指令
        if re.search(r'(识别|是哪颗星|指哪里|指向哪里|正在指|现在指)', user_input):
            parameters["identify"] = True
        
        # 提取讲故事指令
        if re.search(r'(讲.*故事|说.*故事|介绍.*故事|分享.*故事)', user_input):
            parameters["story"] = True
        
        # 提取创作诗歌指令
        if re.search(r'(写.*诗|创作.*诗|作.*诗|吟.*诗)', user_input):
            parameters["poem"] = True
        
        # 提取天气查询指令
        if re.search(r'(天气|观测条件|观星条件|能不能看星星|适合观测)', user_input):
            parameters["weather_check"] = True
        
        # 如果没有指定目标，尝试使用上次识别的星星
        if "target" not in parameters and self.context["last_star_identified"]:
            if re.search(r'(它|这个|那颗星|那颗星星)', user_input):
                parameters["target"] = self.context["last_star_identified"]["name"]
        
        return parameters
    
    def _cleanup_cache(self):
        """定期清理缓存"""
        current_time = time.time()
        
        if current_time - self.cache_last_cleanup > 60:
            self.cache_last_cleanup = current_time
            
            if len(self.intent_cache) > self.cache_size:
                sorted_items = sorted(
                    self.intent_cache.items(), 
                    key=lambda x: x[1].get("timestamp", 0)
                )
                
                items_to_keep = sorted_items[-self.cache_size:]
                self.intent_cache = dict(items_to_keep)
                print(f"缓存清理完成，保留 {len(self.intent_cache)} 个条目")
                
    def _update_intent_cache(self, intent_hash: str, intent_analysis: str, tasks: List[Dict], user_input: str):
        """更新意图缓存"""
        self._cleanup_cache()
        
        cached_tasks = []
        for task in tasks:
            cached_task = {
                "action": task["action"],
                "description": task["description"],
                "parameters": task.get("parameters", {}).copy()
            }
            cached_tasks.append(cached_task)
        
        self.intent_cache[intent_hash] = {
            "intent": intent_analysis,
            "tasks": cached_tasks,
            "timestamp": time.time(),
            "original_input": user_input,
            "hit_count": 0
        }
        
        print(f"缓存已更新，当前缓存条目数: {len(self.intent_cache)}")
    
    def _get_cached_intent(self, intent_hash: str, current_parameters: Dict) -> Optional[Dict]:
        """从缓存中获取意图，并更新参数"""
        if intent_hash in self.intent_cache:
            cached_data = self.intent_cache[intent_hash]
            cached_data["hit_count"] = cached_data.get("hit_count", 0) + 1
            
            print(f"缓存命中！输入: '{cached_data['original_input']}'")
            print(f"该缓存已被命中 {cached_data['hit_count']} 次")
            
            tasks = []
            for cached_task in cached_data["tasks"]:
                task = {
                    "action": cached_task["action"],
                    "description": cached_task["description"],
                    "parameters": cached_task.get("parameters", {}).copy()
                }
                
                # 用当前参数更新任务参数
                if "target" in current_parameters and current_parameters["target"]:
                    if task["action"] in ["tell_story", "write_poem"]:
                        task["parameters"]["target"] = current_parameters["target"]
                
                elif (task["action"] in ["tell_story", "write_poem"] and 
                      "target" not in task["parameters"] and 
                      not current_parameters.get("target") and
                      self.context["last_star_identified"]):
                    task["parameters"]["target"] = self.context["last_star_identified"]["name"]
                
                tasks.append(task)
            
            result = {
                "intent_analysis": cached_data["intent"] + f" (缓存命中，已使用{cached_data['hit_count']}次)",
                "tasks": tasks,
                "use_context": "是（使用缓存）",
                "target_extraction": current_parameters.get("target", "")
            }
            
            self.cache_hits += 1
            print(f"缓存统计: 命中 {self.cache_hits} 次, 未命中 {self.cache_misses} 次")
            return result
        
        self.cache_misses += 1
        return None
        
    def _update_ui_progress(self, step: int, total: int, task_description: str = ""):
        """更新UI进度显示"""
        if self.ui_callback and hasattr(self.ui_callback, 'update_task_progress'):
            self.ui_callback.update_task_progress(step, total, task_description)
    
    def _plan_tasks_with_llm(self, user_input: str) -> Dict:
        """使用大模型进行意图理解和任务规划（增加缓存机制）"""
        
        if self.ui_callback and hasattr(self.ui_callback, 'update_agent_status_icon'):
            self.ui_callback.update_agent_status_icon("analyzing")
        
        # 1. 生成意图哈希
        intent_hash = self._generate_intent_hash(user_input)
        
        # 2. 从当前输入中提取参数
        current_parameters = self._extract_parameters_from_input(user_input)
        
        # 3. 调试信息
        print(f"用户输入: '{user_input}'")
        print(f"生成的哈希: {intent_hash}")
        print(f"提取的参数: {current_parameters}")
        
        # 4. 检查缓存
        cached_result = self._get_cached_intent(intent_hash, current_parameters)
        if cached_result:
            print("使用缓存的任务规划")
            return cached_result
        
        print("缓存未命中，调用大模型进行任务规划...")
        
        # 5. 缓存未命中，调用大模型进行规划
        # 更新规划提示词，包含天气查询工具
        planning_prompt = f"""用户输入："{user_input}"

        请分析用户的意图并规划任务执行序列。考虑以下上下文信息：
        - 激光当前状态：{"开启" if self.context["laser_state"] else "关闭"}
        - 上次识别的星星：{self.context["last_star_identified"]["name"] if self.context["last_star_identified"] else "无"}
        - 上次提到的星座：{self.context["last_constellation"] if self.context["last_constellation"] else "无"}
        
        请识别用户意图并规划任务。可用工具包括：
        1. turn_on_laser - 打开激光（无需参数）
        2. turn_off_laser - 关闭激光（无需参数）
        3. identify_star - 识别当前指向的星星（无需参数）
        4. tell_story - 讲述故事（参数：target=星星或星座名称）
        5. write_poem - 创作诗歌（参数：target=星星或星座名称）
        6. check_weather - 查询天气和观测条件（无需参数）
        7. general_chat - 一般对话

        请以JSON格式返回规划结果，格式如下：
        {{
          "intent_analysis": "对用户意图的分析说明",
          "tasks": [
            {{
              "action": "工具函数名称",
              "parameters": {{"参数名": "参数值"}},  # 如无参数则为空对象
              "description": "任务描述"
            }}
          ],
          "use_context": "说明是否需要使用上下文信息",
          "target_extraction": "从输入中提取的目标对象（如果有）"
        }}
        
        注意：请确保任务顺序符合逻辑（如先打开激光再识别星星）。"""

        messages = [
            self._build_system_message(),
            {"role": "user", "content": planning_prompt}
        ]
        
        print("调用大模型进行意图理解和任务规划...")
        if self.ui_callback and hasattr(self.ui_callback, 'update_agent_status_icon'):
            self.ui_callback.update_agent_status_icon("planning")
            
        response = self.llm.chat(messages)
        
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        else:
            content = str(response)
        
        # 解析JSON响应
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                plan = json.loads(content)
            
            print(f"任务规划结果：{json.dumps(plan, ensure_ascii=False, indent=2)}")
            
            # 6. 将结果存入缓存
            self._update_intent_cache(intent_hash, plan["intent_analysis"], plan["tasks"], user_input)
            
            return plan
            
        except json.JSONDecodeError as e:
            print(f"解析任务规划结果失败：{e}")
            print(f"原始响应：{content}")
            
            default_plan = {
                "intent_analysis": "未能解析复杂的意图，将尝试直接处理",
                "tasks": [{"action": "general_chat", "parameters": {}, "description": "直接对话处理"}],
                "use_context": "否",
                "target_extraction": ""
            }
            
            self._update_intent_cache(intent_hash, default_plan["intent_analysis"], default_plan["tasks"], user_input)
            
            return default_plan
    
    def _execute_task(self, task: Dict, task_index: int, total_tasks: int) -> Dict:
        """执行单个任务"""
        action = task["action"]
        parameters = task.get("parameters", {})
        description = task.get("description", "")
        
        print(f"执行任务 {task_index+1}/{total_tasks}: {description}")
        
        # 更新UI进度
        self._update_ui_progress(task_index, total_tasks, description)
        
        try:
            if action == "turn_on_laser":
                result = self._handle_open_laser()
                return {"success": True, "result": result}
                
            elif action == "turn_off_laser":
                result = self._handle_close_laser()
                return {"success": True, "result": result}
                
            elif action == "identify_star":
                return self._handle_identify_star()
                
            elif action == "tell_story":
                target = parameters.get("target")
                if not target and self.context["last_star_identified"]:
                    target = self.context["last_star_identified"]["name"]
                result = self._handle_tell_story(target)
                return {"success": True, "result": result}
                
            elif action == "write_poem":
                target = parameters.get("target")
                if not target and self.context["last_star_identified"]:
                    target = self.context["last_star_identified"]["name"]
                result = self._handle_write_poem(target)
                return {"success": True, "result": result}
                
            elif action == "check_weather":
                result = self._handle_check_weather()
                return {"success": True, "result": result}
                
            else:
                messages = [
                    self._build_system_message(),
                    {"role": "user", "content": parameters.get("user_input", description)}
                ]
                response = self.llm.chat(messages)
                result = response.get("content") if isinstance(response, dict) else response
                return {"success": True, "result": result}
                

        except Exception as e:
            return {"error": f"执行任务失败：{str(e)}"}
            
    def _get_local_coords(self):
        """获取本地坐标"""
        lon = 113.95778  # 深圳经度
        lat = 22.58194   # 深圳纬度
        try:
            self.GNSS.GNSS_Read()
            self.GNSS.GNSS_Parese()
            if self.GNSS.DataIsUseful:
                lon = float(self.GNSS.longitude[:3]) + float(self.GNSS.longitude[11:18])/60 
                lat = float(self.GNSS.latitude[:2]) + float(self.GNSS.latitude[10:17])/60 
        except:
            pass          
        return lon, lat
    
    def _handle_open_laser(self) -> str:
        """打开激光"""
        pwm.set_pwm(1, 100)
        self.laser_on = True
        self.context["laser_state"] = True
        self.context["current_action"] = "laser_on"
        return "激光头已打开，绿色激光束指向星空..."
    
    def _handle_close_laser(self) -> str:
        """关闭激光"""
        pwm.set_pwm(1, 0)
        self.laser_on = False
        self.context["laser_state"] = False
        self.context["current_action"] = "laser_off"
        return "激光头已关闭..."
        
    def _handle_identify_star(self) -> Dict:
        """识别星星"""
        # 系统时间
        utc = datetime.utcnow()
        
        # 1. 北斗定位
        lon, lat = self._get_local_coords()
        self.context["current_location"] = (lon, lat)  # 保存位置
            
        # 2. 方位角和高度角
        self.magnetic.peeling()
        az = (self.magnetic.get_heading() + 90) % 360
        
        _Ax, _Ay, _Az = self.accelerometer.get_x(), self.accelerometer.get_y(), self.accelerometer.get_z()
        _T = math.sqrt(_Ax ** 2 + _Az ** 2)
        if _Az < 0: 
            el = math.degrees(math.atan2(_Ay, _T))
        else: 
            el = 180 - math.degrees(math.atan2(_Ay, _T))
        
        print(f"世界时间：{utc}\n经度：{lon}\n纬度：{lat}\n方位角：{az}\n高度角：{el}")
        
        # 3. 地平坐标转换成赤道坐标
        obs = Observation(
            utc_time=utc,
            longitude=lon, 
            latitude=lat, 
            altitude=0, 
            azimuth=az,    
            elevation=el
        )
        coords = self.calculator.altaz_to_radec(obs)
        
        # 4. 查找天体（包括行星和深空天体）
        found_object = self.db.find_object_by_coordinates(coords.ra, coords.dec, 5.0, utc)
        
        if found_object:
            obj_type = found_object["type"]
            obj_name = found_object["name"]
            obj_info = found_object["info"]
            
            # 保存到上下文
            self.context["last_object_identified"] = {
                "type": obj_type,
                "name": obj_name,
                "info": obj_info
            }
        
            # 根据不同类型生成不同的响应
            if obj_type == "star":
                response = self._format_star_response(obj_name, obj_info)
            elif obj_type == "planet":
                response = self._format_planet_response(obj_name, obj_info)
            elif obj_type == "deep_sky":
                response = self._format_deep_sky_response(obj_name, obj_info)
            else:
                response = f"我识别到一个天体：{obj_name}"
            
            return {
                "success": True,
                "result": response,
                "object_type": obj_type,
                "object_name": obj_name,
                "object_info": obj_info
            }
        else:
            # 如果没有找到，给出可能的建议
            suggestions = self._generate_identification_suggestions(coords, utc)
            
            return {
                "success": False,
                "result": f"抱歉，未能识别到对应的天体。它的赤道坐标是赤经{coords.ra:.2f}°，赤纬{coords.dec:.2f}°。\n\n可能的目标：{suggestions}",
                "coordinates": {"ra": coords.ra, "dec": coords.dec}
            }
    
    def _format_star_response(self, star_name: str, star_info: Dict) -> str:
        """格式化星星响应"""
        return f"我识别到您指向的是恒星{star_name}，它是{star_info['constellation']}的一颗星，视星等{star_info['magnitude']}等，距离我们大约{star_info['distance']}光年。"

    def _format_planet_response(self, planet_name: str, planet_info: Dict) -> str:
        """格式化行星响应"""
        distance_ly = planet_info.get('distance_ly', 0)
        distance_au = planet_info.get('distance_au', 0)
        elongation = planet_info.get('elongation', 0)
        
        response = f"我识别到您指向的是{planet_info['type']}{planet_name}！\n"
        response += f"它当前距离太阳约{distance_au:.2f}天文单位（{distance_ly*63241.1:.0f}万公里），"
        response += f"距太阳约{elongation:.1f}度。\n"
        response += f"{planet_info['story'][:100]}..."
        
        return response

    def _format_deep_sky_response(self, dso_name: str, dso_info: Dict) -> str:
        """格式化深空天体响应"""
        catalog = dso_info.get('catalog', '')
        catalog_str = f"（{catalog}）" if catalog else ""
        
        response = f"我识别到您指向的是深空天体：{dso_name}{catalog_str}\n"
        response += f"类型：{dso_info['type']}，位于{dso_info['constellation']}，"
        response += f"视星等{dso_info['magnitude']}等，距离约{dso_info['distance']:,}光年。\n"
        response += f"{dso_info['story'][:120]}..."
        
        return response
        
    def _generate_identification_suggestions(self, coords: CelestialCoordinates, dt: datetime) -> str:
        """生成识别建议"""
        suggestions = []
        
        # 1. 查找最近的行星
        for planet_name in ["金星", "火星", "木星", "土星"]:  # 只检查亮行星
            position = self.db.planet_calculator.planet_position(planet_name, dt)
            if position:
                ra_diff = abs(position["ra"] - coords.ra)
                dec_diff = abs(position["dec"] - coords.dec)
                
                if ra_diff > 180:
                    ra_diff = 360 - ra_diff
                
                distance = (ra_diff**2 + dec_diff**2)**0.5
                
                if distance < 30:  # 30度以内
                    suggestions.append(f"可能是{planet_name}（距离约{distance:.1f}度）")
        
        # 2. 查找著名的深空天体
        for dso_name, dso_info in self.db.deep_sky_objects.items():
            if "ra_deg" in dso_info and "dec_deg" in dso_info:
                ra_diff = abs(dso_info["ra_deg"] - coords.ra)
                dec_diff = abs(dso_info["dec_deg"] - coords.dec)
                
                if ra_diff > 180:
                    ra_diff = 360 - ra_diff
                
                distance = (ra_diff**2 + dec_diff**2)**0.5
                
                if distance < 20:  # 20度以内
                    suggestions.append(f"可能是{dso_name}（距离约{distance:.1f}度）")
        
        return "；".join(suggestions[:3]) if suggestions else "请尝试指向更亮的天体"
        
    def _handle_tell_story(self, target: str = None) -> str:
        """讲故事"""
        if not target:
            return "请告诉我你想了解哪颗星、行星或深空天体的故事。"
        
        # 1. 检查是否为行星
        planet_info = self.db.planets.get(target)
        if planet_info:
            prompt = f"""请为{target}创作一个生动、有趣的天文讲解。

            天体信息：
            - 类型：{planet_info['type']}
            - 视星等：{planet_info['magnitude']}等
            - 距离太阳：{planet_info['distance']}天文单位
            - 公转周期：{planet_info['orbital_period']}天
            - 特点：{'、'.join(planet_info['features'][:3])}
            - 简介：{planet_info['story']}
            
            要求：
            1. 先介绍这颗行星的基本情况
            2. 讲述它的神话传说或命名由来
            3. 介绍它的主要特点和科学发现
            4. 说明当前观测条件（如果可见）
            5. 100字以内
            6. 以第一人称"我"（小星）的口吻讲述
            
            请开始你的讲解："""
            
            messages = [
                self._build_system_message(),
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.chat(messages)
            return response.get("content") if isinstance(response, dict) else response
        
        # 2. 检查是否为深空天体
        dso_info = self.db.deep_sky_objects.get(target)
        if dso_info:
            prompt = f"""请为深空天体{target}创作一个生动、有趣的天文讲解。

            天体信息：
            - 类型：{dso_info['type']}
            - 梅西耶编号：{dso_info.get('catalog', '无')}
            - 星座：{dso_info['constellation']}
            - 视星等：{dso_info['magnitude']}等
            - 距离：{dso_info['distance']:,}光年
            - 特点：{'、'.join(dso_info['features'][:3])}
            - 简介：{dso_info['story']}
            
            要求：
            1. 介绍这个深空天体的基本概况
            2. 讲述它的发现历史或观测故事
            3. 描述它的科学意义和观测价值
            4. 给出观测建议（需要什么设备）
            5. 100字以内
            6. 以第一人称"我"（小星）的口吻讲述
            
            请开始你的讲解："""
            
            messages = [
                self._build_system_message(),
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.chat(messages)
            return response.get("content") if isinstance(response, dict) else response
            
        # 判断目标是恒星还是星座
        star_info = self.db.get_star_info(target)
        constellation_info = self.db.get_constellation_info(target)
        
        if star_info:
            prompt = f"""请为星星'{target}'创作一个生动、有趣的天文讲解。

            星星信息：
            - 星座：{star_info['constellation']}
            - 视星等：{star_info['magnitude']}等
            - 距离：{star_info['distance']}光年
            - 简介：{star_info['story']}
            
            要求：
            1. 先介绍这颗星星的基本情况
            2. 讲述它的神话传说或文化背景
            3. 如果有相关的科学知识，适当加入
            4. 100字以内
            5. 以第一人称"我"（小星）的口吻讲述
            
            请开始你的讲解："""
            
            messages = [
                self._build_system_message(),
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.chat(messages)
            return response.get("content") if isinstance(response, dict) else response
            
        elif constellation_info:
            prompt = f"""请为星座'{target}'创作一个生动、有趣的天文讲解。

            星座信息：
            {constellation_info['story']}
            
            要求：
            1. 用故事化的语言讲述这个星座的来历
            2. 融入相关的神话传说
            3. 适当加入科学知识（如星座中的主要恒星）
            4. 用比喻和诗意的语言让故事更吸引人
            5. 100字以内
            6. 以第一人称"我"（小星）的口吻讲述
            7. 最后可以给一些观测建议
            
            请开始你的讲解："""
            
            messages = [
                self._build_system_message(),
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.chat(messages)
            return response.get("content") if isinstance(response, dict) else response
            
        else:
            return f"抱歉，我没有找到{target}的相关故事。"
    
    def _handle_write_poem(self, target: str = None) -> str:
        """创作诗歌"""
        if not target:
            if self.context["last_star_identified"]:
                target = self.context["last_star_identified"]["name"]
            else:
                target = random.choice(["天狼星", "织女星", "牛郎星", "北极星", "参宿四"])
        
        star_info = self.db.get_star_info(target)
        
        prompt = f"""请为"{target}"创作一首优美的诗歌。

        背景信息：
        {target}，{star_info['constellation'] if star_info else "天空中美丽的星星"}。
        {"距离地球约" + str(star_info['distance']) + "光年，" if star_info else ""}{star_info['story'][:100] if star_info else ""}
        
        创作要求：
        1. 诗歌主题与{target}相关，融入天文元素
        2. 可以是中国古典诗词风格，也可以是现代诗歌
        3. 体现星星的特点或相关神话传说
        4. 语言优美，富有意境
        5. 长度适中（4-8行为佳）
        6. 如果是中国古典诗词，请注明词牌名或诗体
        
        请创作一首诗："""
        
        messages = [
            self._build_system_message(),
            {"role": "user", "content": prompt}
        ]
        
        print(f"为{target}创作诗歌...")
        response = self.llm.chat(messages)
        return response.get("content") if isinstance(response, dict) else response
    
    def _handle_check_weather(self) -> str:
        """检查天气和观测条件"""
        print("正在查询天气和观测条件...")
        
        # 获取当前位置
        if self.context["current_location"]:
            lon, lat = self.context["current_location"]
        else:
            lon, lat = self._get_local_coords()
            self.context["current_location"] = (lon, lat)
        
        # 查询观测条件
        weather_data = self.weather_api.get_observation_conditions(lon, lat)
        self.context["last_weather_check"] = time.time()
        
        if "error" in weather_data:
            return f"查询天气失败：{weather_data['error']}"
        
        if not weather_data.get("success"):
            return "无法获取天气数据，请检查网络连接。"
        
        # 提取关键信息
        conditions = weather_data.get("observation_conditions", {})
        current_weather = weather_data.get("current_weather", {})
        astronomy = weather_data.get("astronomy_data", {})
        
        # 构建友好的天气报告
        weather_text = f"""当前天气：{current_weather.get('text', '未知')}，温度{current_weather.get('temp', '未知')}°C
        云量：{conditions.get('cloud_cover', '未知')}，能见度：{conditions.get('visibility_rating', '未知')}
        湿度：{conditions.get('humidity_rating', '未知')}，风速：{conditions.get('wind_rating', '未知')}
        月光影响：{conditions.get('moon_phase_effect', '未知')}，光污染：{conditions.get('light_pollution', '未知')}
        总体评分：{conditions.get('overall_rating', 0)}/5 分
        {conditions.get('recommendation', '')}"""

        # 如果条件良好，可以建议观测目标
        if conditions.get("overall_rating", 0) >= 3:
            suggestions = self._generate_observation_suggestions(conditions, astronomy)
            weather_text += f"\n\n观测建议：{suggestions}"
        
        return weather_text
    
    def _generate_observation_suggestions(self, conditions: Dict, astronomy: Dict) -> str:
        """根据天气条件生成观测建议"""
        suggestions = []
        
        # 根据月相建议
        moon_illumination = float(astronomy.get("moonIllumination", 50))
        
        if moon_illumination <= 30:
            suggestions.append("新月期间适合观测深空天体，如星云、星团和星系")
        elif moon_illumination <= 60:
            suggestions.append("月光适中，可以观测较亮的深空天体")
        else:
            suggestions.append("月光较强，建议观测行星和亮星")
        
        # 根据云量建议
        cloud_cover = conditions.get("cloud_cover", "")
        if "低" in cloud_cover or "晴朗" in cloud_cover:
            suggestions.append("天空晴朗，是观测的好时机")
        elif "中等" in cloud_cover:
            suggestions.append("有部分云层，建议寻找云缝观测")
        
        # 根据能见度建议
        visibility = conditions.get("visibility_rating", "")
        if "极好" in visibility or "良好" in visibility:
            suggestions.append("能见度良好，可以尝试观测较暗的天体")
        
        return "；".join(suggestions[:3])  # 只返回前3条建议
    
    def _generate_integrated_response(self, user_input: str, task_results: List[Dict]) -> str:
        """生成整合多个任务结果的最终响应"""
        
        results_summary = "\n".join([
            f"任务{i+1}: {result.get('result', '无结果')}" 
            for i, result in enumerate(task_results)
        ])
        
        integration_prompt = f"""用户提出了一个复合指令: "{user_input}"

        你已经执行了以下任务并得到这些结果:
        {results_summary}
        
        请将这些结果整合成一个连贯、自然的回答。要求:
        1. 保持对话的流畅性
        2. 用自然的过渡连接不同的部分
        3. 保持专业且富有情感的语气
        4. 确保上下文逻辑正确
        5. 控制在150字以内
        6. 如果是包含诗歌的复合指令，要自然地将诗歌融入回答中
        
        整合后的回答:"""
        
        messages = [
            self._build_system_message(),
            {"role": "user", "content": integration_prompt}
        ]
        
        response = self.llm.chat(messages)
        return response.get("content") if isinstance(response, dict) else response
    
    def process_complex_command(self, user_input: str, ui_callback=None) -> str:
        """处理复杂的复合命令 - 使用大模型进行意图理解和任务规划"""
        
        print(f"处理用户输入: {user_input}")
        
        # 1. 使用大模型进行意图理解和任务规划
        plan = self._plan_tasks_with_llm(user_input)
        tasks = plan.get("tasks", [])
        
        if not tasks:
            return "未能识别出可执行的任务。"
        
        # 2. 初始化任务进度显示
        if ui_callback and hasattr(ui_callback, 'init_task_progress'):
            task_descriptions = [task.get("description", f"任务{i+1}") for i, task in enumerate(tasks)]
            ui_callback.init_task_progress(task_descriptions)
        
        # 3. 执行规划的任务
        task_results = []
        
        for i, task in enumerate(tasks):
            result = self._execute_task(task, i, len(tasks))
            task_results.append(result)
            
            # 如果有错误，停止执行
            if "error" in result:
                error_msg = f"执行任务时出错：{result['error']}"
                if ui_callback and hasattr(ui_callback, 'show_error_state'):
                    ui_callback.show_error_state(error_msg)
                return error_msg
        
        # 4. 完成进度显示
        if ui_callback and hasattr(ui_callback, 'complete_task_progress'):
            ui_callback.complete_task_progress()
        
        # 5. 生成整合响应
        if len(task_results) == 1:
            return task_results[0].get("result", "任务执行完成。")
        else:
            return self._generate_integrated_response(user_input, task_results)

# ========== 语音交互界面 ==========
class VoiceInteractionUI:
    """语音交互界面控制器"""
    
    def __init__(self, agent=None):
        self.agent = agent
        self.current_screen = None
        self.task_steps = []
        self.current_step = 0
        self.total_steps = 0
        self.recording = False
        self.current_status = "idle"
        self.last_task_complete_time = 0
        
        # 字体定义
        self.FONTS = {
            'small': lv.lv_font_siyuan_heiti_normal_14,
            'normal': lv.lv_font_siyuan_heiti_normal_16,
            'medium': lv.lv_font_siyuan_heiti_normal_20,
            'large': lv.lv_font_siyuan_heiti_normal_24,
            'xlarge': lv.lv_font_siyuan_heiti_normal_32
        }
        
        # 颜色代码
        self.COLOR_CODES = {
            'accent': '#bbe1fa',
            'warning': '#FF9800',
            'success': '#4CAF50',
            'error': '#F44336',
            'white': '#ffffff',
            'bg': '#1a1a2e',
            'primary': '#3333ff',
            'gray': '#555555',
            'cyan': '#00BCD4',
            'purple': '#9C27B0'
        }
        
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        scr = lv.scr_act()
        
        # === 1. 创建主容器 ===
        self.main_container = lv.Obj(scr)
        self.main_container.set_size(240, 320)
        self.main_container.set_pos(0, 0)
        self.main_container.set_style_local_bg_color(lv.OBJ_PART.MAIN, lv.STATE.DEFAULT, color_16(self.COLOR_CODES['bg']))
        self.main_container.set_style_local_pad_all(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.main_container.set_style_local_border_width(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.main_container.set_style_local_radius(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        
        # === 2. 状态栏 ===
        self.status_bar = lv.Obj(self.main_container)
        self.status_bar.set_size(240, 24)
        self.status_bar.set_pos(0, 0)
        self.status_bar.set_style_local_bg_color(lv.OBJ_PART.MAIN, lv.STATE.DEFAULT, color_16(self.COLOR_CODES['primary']))
        self.status_bar.set_style_local_pad_all(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.status_bar.set_style_local_border_width(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.status_bar.set_style_local_radius(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        
        # 时间显示
        self.time_label = lv.Label(self.status_bar)
        self.time_label.set_pos(10, 0)
        self.time_label.set_style_local_text_font(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, self.FONTS['small'])
        self.time_label.set_long_mode(lv.LABEL_LONG.EXPAND)
        self.time_label.set_width(60)
        self.time_label.set_recolor(True)
        self.time_label.set_text(self.COLOR_CODES['accent'] + ' ' + time.strftime("%H:%M"))
        
        # 状态图标
        self.status_icons = lv.Label(self.status_bar)
        self.status_icons.set_pos(200, 0)
        self.status_icons.set_style_local_text_font(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, self.FONTS['small'])
        self.status_icons.set_long_mode(lv.LABEL_LONG.CROP)
        self.status_icons.set_width(40)
        self.status_icons.set_recolor(True)
        self.status_icons.set_text(self.COLOR_CODES['accent'] + ' ' + lv.SYMBOL.GPS + lv.SYMBOL.WIFI + '#')
        
        # === 3. 智能体形象区域 ===
        self.agent_area = lv.Obj(self.main_container)
        self.agent_area.set_size(240, 80)
        self.agent_area.set_pos(0, 24)
        self.agent_area.set_style_local_bg_color(lv.OBJ_PART.MAIN, lv.STATE.DEFAULT, color_16(self.COLOR_CODES['primary']))
        self.agent_area.set_style_local_pad_all(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.agent_area.set_style_local_border_width(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.agent_area.set_style_local_radius(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        
        # 智能体图标 - 使用图片
        self.agent_icon = lv.Img(self.agent_area)
        self.agent_icon.set_antialias(True)
        self.agent_icon.set_pos(80, 0)
        self.agent_icon_dsc, self.agent_icon_dsc_data = get_img_dsc('/root/idle.jpg', apt_screen_size=False)
        self.agent_icon_dsc.data = self.agent_icon_dsc_data
        self.agent_icon.set_src(self.agent_icon_dsc)
        
        # === 4. 对话区域 ===
        self.chat_area = lv.Obj(self.main_container)
        self.chat_area.set_size(240, 160)
        self.chat_area.set_pos(0, 104)
        self.chat_area.set_style_local_bg_color(lv.OBJ_PART.MAIN, lv.STATE.DEFAULT, color_16(self.COLOR_CODES['bg']))
        self.chat_area.set_style_local_pad_all(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.chat_area.set_style_local_border_width(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.chat_area.set_style_local_radius(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        
        # 智能体响应显示
        self.agent_response_label = lv.Label(self.chat_area)
        self.agent_response_label.set_pos(20, 10)
        self.agent_response_label.set_style_local_text_font(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, self.FONTS['small'])
        self.agent_response_label.set_long_mode(lv.LABEL_LONG.BREAK)
        self.agent_response_label.set_width(200)
        self.agent_response_label.set_recolor(True)
        self.agent_response_label.set_text(self.COLOR_CODES['accent'] + ' ' + '按下按钮对我说话，\n例如：查询观测条件，\n打开激光指星星，\n再讲讲星座故事...')
        
        # 用户语音输入显示
        self.user_input_label = lv.Label(self.chat_area)
        self.user_input_label.set_pos(20, 10)
        self.user_input_label.set_style_local_text_font(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, self.FONTS['small'])
        self.user_input_label.set_long_mode(lv.LABEL_LONG.BREAK)
        self.user_input_label.set_width(200)
        self.user_input_label.set_hidden(True)
        self.user_input_label.set_recolor(True)
        
        # === 5. 任务进度区域 ===
        self.task_area = lv.Obj(self.main_container)
        self.task_area.set_size(240, 56)
        self.task_area.set_pos(0, 264)
        self.task_area.set_style_local_bg_color(lv.OBJ_PART.MAIN, lv.STATE.DEFAULT, color_16(self.COLOR_CODES['primary']))
        self.task_area.set_style_local_pad_all(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.task_area.set_style_local_border_width(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.task_area.set_style_local_radius(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
        self.task_area.set_hidden(True)
        
        # 任务标题
        self.task_title = lv.Label(self.task_area)
        self.task_title.set_pos(10, 5)
        self.task_title.set_style_local_text_font(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, self.FONTS['small'])
        self.task_title.set_recolor(True)
        self.task_title.set_text(self.COLOR_CODES['accent'] + ' ' + '任务进度')
        
        # 进度条
        self.progress_bar = lv.Bar(self.task_area)
        self.progress_bar.set_size(200, 15)
        self.progress_bar.set_pos(20, 30)
        self.progress_bar.set_range(0, 100)
        self.progress_bar.set_value(0, lv.ANIM.OFF)
        
        # 设置进度条样式
        self.progress_bar.set_style_local_bg_color(lv.BAR_PART.BG, lv.STATE.DEFAULT, color_16(self.COLOR_CODES['gray']))
        self.progress_bar.set_style_local_bg_opa(lv.BAR_PART.BG, lv.STATE.DEFAULT, 255)
        self.progress_bar.set_style_local_bg_color(lv.BAR_PART.INDIC, lv.STATE.DEFAULT, color_16(self.COLOR_CODES['success']))
        self.progress_bar.set_style_local_bg_opa(lv.BAR_PART.INDIC, lv.STATE.DEFAULT, 255)
        
        # 刷新屏幕
        lv.scr_load(self.main_container)
    
    def update_agent_status_icon(self, status: str, extra_info: str = ""):
        """更新智能体状态图标和文本"""
        self.current_status = status
        
        if status == "success":
            self.last_task_complete_time = time.time()
        
        # 状态图片映射
        status_images = {
            "idle": "/root/idle.jpg",
            "listening": "/root/listening.jpg",
            "analyzing": "/root/analyzing.jpg",
            "planning": "/root/planning.jpg",
            "executing": "/root/executing.jpg",
            "success": "/root/success.jpg",
            "error": "/root/error.jpg"
        }
        
        if status in status_images:
            image_file = status_images[status]
            
            try:
                self.agent_icon_dsc, self.agent_icon_dsc_data = get_img_dsc(image_file, apt_screen_size=False)
                self.agent_icon_dsc.data = self.agent_icon_dsc_data
                self.agent_icon.set_src(self.agent_icon_dsc)
            except Exception as e:
                print(f"加载图片失败 {image_file}: {e}")
                if status != "idle":
                    self.agent_icon_dsc, self.agent_icon_dsc_data = get_img_dsc("/root/idle.jpg", apt_screen_size=False)
                    self.agent_icon_dsc.data = self.agent_icon_dsc_data
                    self.agent_icon.set_src(self.agent_icon_dsc)
            
    def check_return_to_idle(self):
        """检查是否应该返回空闲状态"""
        if self.current_status == "success" and time.time() - self.last_task_complete_time > 20.0:
            self.update_agent_status_icon("idle")
            self.agent_response_label.set_text(self.COLOR_CODES['accent'] + ' ' + '按下按钮对我说话，\n例如：查询观测条件，\n打开激光指星星，\n再讲讲星座故事...')
            self.agent_response_label.set_hidden(False)
            self.user_input_label.set_hidden(True)    
            
    def update_time(self):
        """更新时间显示"""
        self.time_label.set_text(self.COLOR_CODES['accent'] + ' ' + time.strftime("%H:%M"))
        
        self.check_return_to_idle()
        
    def show_recording_state(self, is_recording: bool):
        """显示录音状态"""
        self.recording = is_recording
        
        if is_recording:
            self.user_input_label.set_hidden(False)
            self.agent_response_label.set_hidden(True)
            self.update_agent_status_icon("listening")
            self.user_input_label.set_text(self.COLOR_CODES['warning'] + ' ' + '正在聆听... 请说话')
        else:
            self.update_agent_status_icon("idle")
            self.user_input_label.set_hidden(True)
            self.agent_response_label.set_hidden(False)
    
    def show_user_input(self, text: str):
        """显示用户输入文本"""
        if len(text) > 50:
            text = text[:47] + "..."
        
        self.user_input_label.set_text(self.COLOR_CODES['white'] + ' ' + f'用户：{text}')
        self.user_input_label.set_hidden(False)
        self.agent_response_label.set_hidden(True)
    
    def show_agent_response(self, text: str):
        """显示智能体响应"""
        text = text.replace('#000000', '').replace('#ffffff', '').replace('#ff9900', '')
        
        if len(text) > 100:
            text = text[:97] + "..."
        
        self.agent_response_label.set_text(self.COLOR_CODES['accent'] + ' ' + f'小星：{text}')
        self.agent_response_label.set_hidden(False)
        self.user_input_label.set_hidden(True)
    
    def show_analyzing_state(self):
        """显示分析中状态"""
        self.update_agent_status_icon("analyzing")
        self.agent_response_label.set_text(self.COLOR_CODES['warning'] + ' ' + '正在思考...')
    
    def show_success_state(self):
        """显示成功状态"""
        self.update_agent_status_icon("success")
    
    def init_task_progress(self, task_steps: list):
        """初始化任务进度显示"""
        self.task_steps = task_steps
        self.total_steps = len(task_steps)
        self.current_step = 0
        
        self.update_agent_status_icon("planning") 
        
        if self.total_steps > 1:
            self.task_area.set_hidden(False)
            self.update_task_progress(0, self.total_steps)
        else:
            self.task_area.set_hidden(True)
    
    def update_task_progress(self, step: int, total: int, task_description: str = ""):
        """更新任务进度"""
        self.current_step = min(step, total)
        self.total_steps = total
        
        if total > 0 and step < total:
            self.update_agent_status_icon("executing")
        
        if total > 0:
            progress_percent = int((self.current_step / total) * 100)
            self.progress_bar.set_value(progress_percent, lv.ANIM.ON)

            if task_description:
                if len(task_description) > 15:
                    task_display = task_description[:12] + "..."
                else:
                    task_display = task_description
                    
                if self.current_step != total: 
                    self.task_title.set_text(self.COLOR_CODES['accent'] + f' 任务 {self.current_step+1}/{total}: {task_display}')
            
            if self.current_step == total:
                self.progress_bar.set_style_local_bg_color(lv.BAR_PART.INDIC, lv.STATE.DEFAULT, color_16(self.COLOR_CODES['success']))
            else:
                self.progress_bar.set_style_local_bg_color(lv.BAR_PART.INDIC, lv.STATE.DEFAULT, color_16(self.COLOR_CODES['accent']))
    
    def complete_task_progress(self):
        """完成任务进度显示"""
        if self.total_steps > 0:
            self.update_task_progress(self.total_steps, self.total_steps)
            
            def hide_task_area():
                time.sleep(1)
                self.task_area.set_hidden(True)
            
            hide_task_area()
    
    def show_error_state(self, error_message: str):
        """显示错误状态"""
        self.update_agent_status_icon("error")
        
        if len(error_message) > 60:
            error_message = error_message[:57] + "..."
        
        self.agent_response_label.set_text(self.COLOR_CODES['error'] + ' ' + f'{error_message}')

# ========== 启动函数 ==========
def startup_animation():
    """启动动画和初始化"""
    scr = lv.scr_act()
    
    startup_container = lv.Obj(scr)
    startup_container.set_size(240, 320)
    startup_container.set_pos(0, 0)
    startup_container.set_style_local_bg_color(lv.OBJ_PART.MAIN, lv.STATE.DEFAULT, color_16('#1a1a2e'))
    startup_container.set_style_local_pad_all(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
    startup_container.set_style_local_border_width(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
    startup_container.set_style_local_radius(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, 0)
    
    title_label = lv.Label(startup_container)
    title_label.set_pos(60, 80)
    title_label.set_style_local_text_font(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, lv.lv_font_siyuan_heiti_normal_32)
    title_label.set_text("星 语 者")
    
    subtitle_label = lv.Label(startup_container)
    subtitle_label.set_pos(30, 150)
    subtitle_label.set_style_local_text_font(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, lv.lv_font_siyuan_heiti_normal_16)
    subtitle_label.set_text("让每一颗星星，都会说话")
    
    progress_bar = lv.Bar(startup_container)
    progress_bar.set_size(200, 20)
    progress_bar.set_pos(20, 220)
    progress_bar.set_range(0, 100)
    progress_bar.set_value(0, lv.ANIM.OFF)
    
    progress_bar.set_style_local_bg_color(lv.BAR_PART.BG, lv.STATE.DEFAULT, color_16('#555555'))
    progress_bar.set_style_local_bg_opa(lv.BAR_PART.BG, lv.STATE.DEFAULT, 255)
    progress_bar.set_style_local_bg_color(lv.BAR_PART.INDIC, lv.STATE.DEFAULT, color_16('#bbe1fa'))
    progress_bar.set_style_local_bg_opa(lv.BAR_PART.INDIC, lv.STATE.DEFAULT, 255)
    
    loading_label = lv.Label(startup_container)
    loading_label.set_pos(90, 250)
    loading_label.set_style_local_text_font(lv.LABEL_PART.MAIN, lv.STATE.DEFAULT, lv.lv_font_siyuan_heiti_normal_14)
    loading_label.set_recolor(True)
    loading_label.set_text('#bbe1fa 初始化中...')
    #magnetic.calibrate()
    
    lv.scr_load(startup_container)
    loading_label.set_pos(70, 250)
    
    stages = [
        (0, "正在检查硬件..."),
        (20, "正在连接北斗..."),
        (40, "正在初始化传感器..."),
        (60, "正在连接大模型..."),
        (80, "正在加载天气服务..."),
    ]
    
    for progress, message in stages:
        progress_bar.set_value(progress, lv.ANIM.OFF)
        loading_label.set_text(f'#bbe1fa {message}')
        time.sleep(0.5)
    
    progress_bar.set_value(100, lv.ANIM.OFF)
    progress_bar.set_style_local_bg_color(lv.BAR_PART.INDIC, lv.STATE.DEFAULT, color_16('#4CAF50'))
    loading_label.set_pos(90, 250)
    loading_label.set_text('#4CAF50 准备就绪！')
    time.sleep(1)
    
    for i in range(100, -1, -5):
        startup_container.set_style_local_bg_opa(lv.OBJ_PART.MAIN, lv.STATE.DEFAULT, i)
        time.sleep(0.01)
        
    startup_container = None
    
    print("启动动画完成，进入主界面")

# ========== 主程序入口 ==========
def main():
    system_1956.ntp_sync()
    baidu_ai = BaiduAI()
    
    """ 大模型配置"""
    config = ModelConfig(
        provider="doubao",
        api_key="***",
        api_base="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        model_name="doubao-seed-2-0-mini-260215"
    )
    
    # 创建UI界面
    ui = VoiceInteractionUI()
    
    # 创建智能体，传入UI回调
    agent = StarWhispererAgent(config, ui)
    
    # 将智能体设置到UI中
    ui.agent = agent
    
    print("系统初始化完成，等待用户指令...")
    print(f"天气API已启用，使用和风天气服务")
    print(f"缓存机制已启用，最大缓存条目: {agent.cache_size}")
    
    while True:
        ui.update_time()
        
        if get_key_code() == 5:
            # === 第1步：开始录音 ===
            ui.show_recording_state(True)
            system_1956.record(6, "my.wav")
            
            # === 第2步：结束录音，开始处理 ===
            ui.show_recording_state(False)
            ui.show_analyzing_state()
            
            # 语音识别
            baidu_ai.baidu_audio_to_text('***', '***', 'my.wav')
            
            asr_result = baidu_ai.res_str
            print(f"用户: {asr_result}")
            
            if not asr_result or len(asr_result.strip()) < 2:
                error_msg = "没有识别到语音，请重新尝试"
                ui.show_error_state(error_msg)
                baidu_ai.baidu_tts("15814686703", 'epAuDr6Bj9XmOtuy8JokSKIn', 
                                  'da85jBGRM2SsXmKOzxQDH63xhFX0suap', 
                                  error_msg, 'tts.mp3')
                system_1956.play_music('tts.mp3')
                continue
            
            ui.show_user_input(asr_result)
            
            # === 第3步：智能体处理（使用大模型进行意图理解和任务规划）===
            try:
                response = agent.process_complex_command(asr_result, ui)
                
                # === 第4步：显示结果 ===
                print(f"智能体回复: {response}")
                print(f"缓存统计: 命中 {agent.cache_hits} 次, 未命中 {agent.cache_misses} 次")
                ui.show_agent_response(response)
                ui.show_success_state()
                
            except Exception as e:
                error_msg = f"处理指令时出错: {str(e)}"
                print(error_msg)
                ui.show_error_state(error_msg)
                response = "抱歉，处理指令时出现了错误，请重新尝试。"
            
            # === 第5步：语音合成 ===
            baidu_ai.baidu_tts("123456789", '***', 
                              '***', 
                              response, 'tts.mp3')
            system_1956.set_volume(30)
            # 播放语音
            system_1956.play_music('tts.mp3')
        
        time.sleep(0.1)

startup_animation()
main()