---
name: portrait-shadow-relight
description: 人像面部阴影美化。当用户要求去除或减弱脸上的树影/枝叶斑驳光影/硬阴影，提亮脸部，让肤色均匀通透，同时保持人物身份、姿势、发型、服饰和构图不变时使用。适用于户外逆光或侧光、阴影斑驳、脸部偏暗偏灰的人像照片。不适用于换脸、换背景、大幅度瘦脸或改变五官。
---

# 人像面部阴影美化（Portrait Shadow Relight）

## 目标

把"脸上有斑驳硬阴影、肤色偏暗偏灰、眼神发闷"的原片，变成"光线干净柔和、肤色通透红润、眼睛有神"的成片。
**只改光和色，不改人。** 人物身份、五官形状、表情、姿势、发型走向、配饰、服装、构图、镜头视角全部保持原样。

## 典型输入与输出（参考案例）

| 维度 | 原图特征 | 目标效果 |
|---|---|---|
| 面部光影 | 枝叶投影形成网状斑驳阴影，覆盖脸颊、鼻梁、额头，明暗断裂 | 阴影图案完全消除，只保留自然的立体过渡（鼻翼、下颌的柔和明暗） |
| 整体曝光 | 脸部欠曝，对比偏硬 | 脸部提亮约 +0.5 ~ +1 EV，对比变柔和 |
| 肤色 | 偏灰、偏黄、缺少血色 | 白里透红，双颊淡淡红润，肤色统一 |
| 眼睛 | 眼窝在阴影里，眼神较暗 | 眼睛提亮，虹膜更清晰通透，保留自然高光 |
| 嘴唇 | 暗红偏闷 | 提高明度和饱和度，呈清透的玫瑰红/粉红，边缘自然 |
| 皮肤质感 | 有毛孔和细小瑕疵 | 轻度磨皮，细节保留约 70%，不能出现塑料感 |
| 头发 | 整体偏暗，高光不明显 | 暖色高光，发丝层次更清晰，发丝形状不变 |
| 背景 | 山林偏棕、偏灰、饱和度低 | 秋叶变为明亮的金橙色，饱和度和明度明显提升；天空更纯净、通透的蓝 |
| 衣物/配饰 | 偏暗 | 亮度提升，质感（针织纹理、纱质蝴蝶结、项链）保留 |

## 工作流程

按顺序执行，不要跳步。

### 第 1 步：分析原图
先输出简短分析（内部使用即可，不必全部展示给用户）：
1. 光源方向、硬度（直射/散射），阴影来源（树枝、栏杆、帽檐等）
2. 阴影覆盖的面部区域：额头、眼窝、鼻梁、颊部、下巴、颈部
3. 必须保留的自然结构阴影：鼻翼下、下颌线、颈部投影、耳后
4. 必须原样保留的元素：发饰、项链、衣服纹理、背景轮廓
5. 原图风险点：逆光轮廓、发丝穿插到脸上的阴影（不要误当作斑驳阴影抹掉）

### 第 2 步：区域划分
把图像分成独立处理的区域，各区域参数不同：
- **A 面部皮肤**：重点处理，去阴影 + 匀肤 + 提亮
- **B 五官**：眼睛、眉毛、嘴唇，局部提亮和增色
- **C 头发**：只做暖色高光和轻微提亮
- **D 衣物与配饰**：轻度提亮，保留纹理
- **E 背景**：独立调色，提高饱和度和明度
- **F 天空**：压缩杂色，提纯蓝色

### 第 3 步：去除斑驳阴影（核心步骤）
- 目标：消除"图案状"阴影，而不是把整张脸压平。
- 做法：把阴影区域的亮度向周围受光区域拉平，同时保留脸部低频立体结构（颧骨、鼻梁、下颌的大块明暗）。
- 阴影边界必须羽化，禁止出现"补丁感""贴图感"或颜色断层。
- 阴影区域的色温要和受光区域一致：阴影常偏蓝灰，需要回补暖色。
- 保留皮肤纹理：不允许用纯模糊覆盖，纹理必须从邻近受光区域延续。

### 第 4 步：整体提亮与肤色
- 肤色目标：自然偏暖的亮肤色，带淡淡的粉红血色，脸颊、鼻尖、耳廓略有红润。
- 不要出现整体发白、油光、过曝；高光区域保持有皮肤细节。
- 全脸肤色必须统一：额头、脸颊、鼻子、下巴、颈部不能出现色差。

### 第 5 步：五官微调
- 眼睛：提亮眼白和虹膜，增强通透感，保留原有眼神方向和眼形，不放大眼睛。
- 嘴唇：提高明度与饱和度，清透玫瑰红，保持原唇形与微笑弧度。
- 眉毛：轻微提升清晰度即可。

### 第 6 步：头发
- 加入暖色高光（偏棕/栗色），让发丝在阳光下有光泽。
- 不改变发型、发丝走向、飞发和蝴蝶结位置。

### 第 7 步：背景与天空
- 山林：提升饱和度和明度，偏金橙色的秋色，保持原有山体轮廓和虚化程度。
- 天空：更干净通透的蓝，不要出现色带或过度渐变。
- 背景调色幅度应与人物协调，不要比人物更抢眼。

### 第 8 步：质检（必做）
逐项核对，任何一项不通过就回到对应步骤重做：
- [ ] 脸上不再有任何树影或斑驳图案
- [ ] 脸部仍有立体感，没有变成"一张平面"
- [ ] 人物身份可识别，与原片是同一个人（五官比例、脸型、眼距、嘴形不变）
- [ ] 表情、头部角度、视线、姿势没有变化
- [ ] 皮肤有可见的毛孔级质感，无塑料感、无蜡像感
- [ ] 发饰、项链、针织衣纹理完整
- [ ] 肤色统一，没有色块、没有边界痕迹
- [ ] 背景色与肤色协调，整体色调统一
- [ ] 无多余手指、边缘伪影、扭曲或重影
- [ ] 分辨率和画幅与原图一致

## 给 AI 大模型的指令模板

### 通用提示词（正向）

```
Retouch this portrait photo. Keep the person's identity, facial features,
expression, pose, hairstyle, hair accessory, necklace, clothing and composition
exactly unchanged.

Main task: completely remove the dappled, leaf-shaped hard shadows from the
face, forehead, cheeks and nose. Keep only soft, natural facial contouring
(nose-wing shadow, jawline, under-chin). The face should look evenly lit by
soft natural sunlight.

Skin: brighten the face by about half to one stop, make the tone bright, warm
and healthy with a subtle rosy blush on the cheeks. Unify the skin color
across face and neck. Light skin smoothing only, keep visible natural skin
texture.

Eyes: brighten and clarify the eyes, keep original shape and gaze direction.
Lips: make them fresher and more saturated, a clear rose-pink, keep the
original lip shape and smile.

Hair: add warm golden-brown highlights and slightly lift the shadows, keep all
strands and the flyaway hairs.

Background: increase saturation and brightness of the autumn mountain to a
vivid golden-orange, keep the same mountain shape and blur. Make the sky a
clean, clear blue.

Style: natural, realistic, bright, clean, soft sunlight, photographic.
```

### 负向提示词

```
changed face, different person, face reshaping, bigger eyes, slimmer face,
altered smile, plastic skin, waxy skin, over-smoothed, blurry skin, flat
lighting, overexposed, blown highlights, color patches, visible mask edges,
residual leaf shadow, uneven skin tone, oversaturated skin, orange skin,
cartoon, illustration, extra fingers, distorted hair, missing accessory,
changed background shape, text, watermark
```

### 如果工具支持分区重绘（inpainting）

1. 只对面部皮肤区域建立蒙版，蒙版边缘羽化 8~20 px。
2. 重绘强度（denoising strength）：0.25 ~ 0.4。高于 0.5 容易改变五官。
3. 加载人脸身份保持（如 IP-Adapter FaceID、InstantID 等）以锁定身份。
4. 背景和天空单独调色，不要与面部使用同一次重绘。
5. 最后用轻度细节增强（锐化 / 纹理恢复）补回皮肤质感。

### 参数参考（传统修图软件 / 调色流程）

| 区域 | 参数 |
|---|---|
| 面部 | 阴影 +40 ~ +70，曝光 +0.5 ~ +0.8，对比 −10 ~ −20，色温偏暖 +3 ~ +8，色调偏洋红 +2 ~ +5 |
| 磨皮 | 频率分离，低频柔化，高频保留 60~80% |
| 嘴唇 | 饱和度 +15 ~ +25，明度 +8 ~ +12 |
| 眼睛 | 局部曝光 +0.3，清晰度 +10 |
| 头发 | 暖色高光，阴影轻提 +10 ~ +20 |
| 背景（橙黄色段） | 饱和度 +30 ~ +50，明度 +10 ~ +20 |
| 天空（蓝色段） | 饱和度 +10 ~ +15，明度 +5 |

## 边界与注意事项

- 不换脸、不改变身份、不瘦脸、不放大眼睛、不改变年龄感和种族特征。
- 美化幅度应保持"自然、像好的摄影后期"，避免过度磨皮和过度美白。
- 如果原图本身不存在斑驳阴影，则只做轻度提亮与调色，不要强行"处理"。
- 如果面部被遮挡、模糊或分辨率太低，先说明限制，不要凭空生成不存在的细节。
- 输出成片如经过 AI 美化，分享时建议如实标注"经过后期美化"。
- 用户要求"更自然"时，把所有强度参数降低约 30%；要求"更精致/更明显"时，提高约 30%，但仍然不得越过身份保持的底线。

## 可扩展方向

- **强度档位**：natural（轻度）/ standard（标准，对应参考案例）/ glam（强度）
- **光线风格**：保持阳光感 / 柔光箱感 / 黄金时刻暖光
- **场景扩展**：窗帘/百叶窗影、帽檐阴影、眼镜框阴影、逆光脸部欠曝
- **批量处理**：同一组照片使用同一套参数，保证肤色和背景色调一致
