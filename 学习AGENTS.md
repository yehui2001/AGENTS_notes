---
planner:
  log:
    - start: 2026-09-18 15:21:46
---
## 学习计划

| 章节                 | 建议优先级  | 学习方式               | 原因                                         | 完成  |
| :----------------- | :----- | :----------------- | :----------------------------------------- | --- |
| 第1章 初识智能体          | 高      | 精读                 | 建立 Agent、环境、感知、行动、目标的基本模型                  | ✅   |
| 第2章 发展史            | 低      | 快速浏览               | 建立历史脉络即可                                   | ✅   |
| 第3章 LLM 基础         | 中      | 选择性阅读              | 理解 token、上下文、提示、Transformer 和模型局限；首轮不必推导公式 |     |
| 第4章 经典范式           | **最高** | 精读并重写代码            | ReAct、Plan-and-Solve、Reflection 是后续基础      | ✅   |
| 第5章 低代码平台          | 低至中    | 选一个体验              | 了解产品形态，不必同时掌握 Coze、Dify、n8n                |     |
| 第6章 框架实践           | 中      | 横向比较，选一个           | 了解框架帮你封装了什么，不要同时深学所有框架                     |     |
| 第7章 自建 Agent 框架    | **最高** | 精读并动手实现            | 理解模型适配、消息、Agent 基类、工具注册和运行循环               |     |
| 第8章 记忆与检索          | **最高** | 精读，做 RAG 项目        | 解决外部知识、长期记忆和事实依据问题                         |     |
| 第9章 上下文工程          | **最高** | 精读并实验              | 决定长任务稳定性、成本和信息利用率                          |     |
| 第10章 通信协议          | 高      | MCP 实做；A2A/ANP 先理解 | MCP 已很实用，多 Agent 协议可后置                     |     |
| 第11章 Agentic-RL    | 后置     | 第二轮再学              | 属于模型训练方向，入门应用开发暂时用不到                       |     |
| 第12章 性能评估          | **最高** | 提前学习并建立测试集         | 没有评测的 Agent 只能算演示                          |     |
| 第13章 旅行助手          | 高      | 作为综合项目参考           | 包含数据模型、多 Agent、MCP 和前后端                    |     |
| 第14章 Deep Research | 高      | 与第13章二选一           | 非常适合练搜索、筛选、引用和报告生成                         |     |
| 第15章 赛博小镇          | 低      | 兴趣扩展               | 项目有趣，但涉及游戏和社会模拟，主线较远                       |     |
| 第16章 毕业设计          | 高      | 用自己的题目完成           | 检验能否独立设计完整系统                               |     |




## 经典智能体范式构建

### ReAct范式

Thought-Action-Observation 的循环

思想：**推理使得行动更具目的性，而行动则为推理提供了事实依据。**

![4-1.png|775](https://raw.githubusercontent.com/yehui2001/imgbed/main/4-1.png)

$$
(h_t, a_t) = \pi\left(q, (a_1, o_1), \ldots, (a_{t-1}, o_{t-1})\right)
$$
在每个时间步$t$中 ，大模型$\pi$ 会根据初始问题$q$ 以及，之前所有步骤的“行动，观测”历史轨迹，来生成当前的思考$th_t$和行动$a_t$ ，并通过工具来获得当前的观测结果$o_t$
$$
o_t= T(a_t)
$$

AGENTS 与LLM之间交互的规范：
- **角色定义**： “你是一个有能力调用外部工具的智能助手”，设定了LLM的角色。
- **工具清单 (`{tools}`)**： 告知LLM它有哪些可用的“手脚”。
- **格式规约 (`Thought`/`Action`)**： 这是最重要的部分，它强制LLM的输出具有结构性，使我们能通过代码精确解析其意图。
- **动态上下文 (`{question}`/`{history}`)**： 将用户的原始问题和不断累积的交互历史注入，让LLM基于完整的上下文进行决策。

具体细节详见 `/home/yehui/Documents/hello-agents/ReAct.py`


### Plan-and-Solve

其<mark style="background: #FFB8EBA6;">核心动机</mark>是为了解决思维链在处理多步骤、复杂问题时容易“偏离轨道”的问题。

思想：
1. **规划阶段 (Planning Phase)**： 首先，智能体会接收用户的完整问题。它的第一个任务不是直接去解决问题或调用工具，而是**将问题分解，并制定出一个清晰、分步骤的行动计划**。这个计划本身就是一次大语言模型的调用产物。
2. **执行阶段 (Solving Phase)**： 在获得完整的计划后，智能体进入执行阶段。它会**严格按照计划中的步骤，逐一执行**。每一步的执行都可能是一次独立的 LLM 调用，或者是对上一步结果的加工处理，直到计划中的所有步骤都完成，最终得出答案

具体的形式化表达：
首先，规划模型$\pi_{\text{plan}}$根据原始问题 $q$ 生成一个包含$n$ 个步骤的计划 $P = (p_1, p_2, \dots, p_n)$：
$$
P=\pi_{plan(q)}
$$
在执行阶段$\pi_{solve}$会逐步完成计划中的各个步骤。对于第 $i$ 个步骤，其解决方案 $s_i$ 的生成会同时依赖于原始问题 $q$、完整计划 $P$ 以及之前所有步骤的执行结果$(s_1,…,s_{i−1})$：
最终答案就是最后一个步骤的执行结果$s_n$


![r-4-2.png|825](https://raw.githubusercontent.com/yehui2001/imgbed/main/4-2.png)

### Reflection

**思想**：执行$\rightarrow$ 反思$\rightarrow$ 优化

执行： Agents先用我们熟悉的方法(ReAct或Plan-and-Solve) 尝试完成任务，生成一个初步的解决方案，看作<mark style="background: #FFB86CA6;">“初稿”</mark>

反思： 智能体会调用一个独立的、或者带有特殊提示词的大语言模型实例，会审视第一步生成的”初稿“，并展开多维度测评，会生成一段结构性的<mark style="background: #FFB86CA6;">”反馈“</mark>。

优化：智能体将<mark style="background: #FFB86CA6;">“初稿”</mark>和<mark style="background: #FFB86CA6;">“反馈”</mark>作为新的上下文，再次调用大语言模型，要求它根据反馈内容对初稿进行修正，生成一个更完善的“修订稿”。

$O_i$代表第$i$次迭代产生的输出，反思模型根据当前的输出生成对应反馈$F_{i}:$
$$
F_{i} = \pi_{reflect}(Task,O_i)
$$
优化模型又会结合原始任务、上一版的输出以及反馈，生成新一版的输出$O_{i+1}:$
$$
O_{i+1} = \pi_{refine}(Task,O_{i},F_{i})
$$
![c-4-3.png|600](https://raw.githubusercontent.com/yehui2001/imgbed/main/4-3.png)

`Trajectory`：当前任务从开始到现在的完整行动轨迹，只针对于当前任务。
`Evaluator`: 的作用是评估当前任务轨迹，而不是直接执行任务。
`Self-reflection:` 总结为什么出问题、以后怎么避免 $\rightarrow$ `Experience`  得到经验


成本效益分析：

（1） 成本
1.**模型调用开销增加**，因为每进行一轮迭代，至少需要额外调用两次大语言模型(一次用于反思，一次用于优化)

2.**任务延迟显著提高**，Reflection是一个串行的过程，每一轮的优化都必须等待上一轮反思完成。

（2）效益
1.**解决方案质量的跃迁**

2.**鲁棒性和可靠性增强**

综上所述，Reflection 机制是一种典型的“以成本换质量”的策略。它非常适合那些**对最终结果的质量、准确性和可靠性有极高要求，且对任务完成的实时性要求相对宽松**的场景






##  Agent框架构建


一个框架的本质，是提供一套经过验证的“规范”。它将所有智能体共有的、重复性的工作（如主循环、状态管理、工具调用、日志记录等）进行抽象和封装，让我们在构建新的智能体时，能够专注于其独特的业务逻辑，而非通用的底层实现。

优势：
1.**提升代码复用与开发效率**
2.**实现核心组件的解耦与可扩展性**
3.**标准化复杂的状态管理**：在真实的、长时运行的智能体应用中，状态管理需要处理上下文窗口限制、历史信息持久化、多轮对话状态跟踪等问题。
4.**简化可观测性与调试过程**： 例如引入事件回调机制(Callbacks)，可以在智能体生命周期的关键节点自动触发日志记录或数据上报

### 四种智能体框架对比

![image.png](https://raw.githubusercontent.com/yehui2001/imgbed/main/20260918152129108.png)

**LangGraph**：作为 LangChain 生态的扩展，LangGraph 另辟蹊径，将智能体的执行流程建模为**图 (Graph)**[4]。在传统的链式结构中，信息只能单向流动。而 LangGraph 将每一步操作（如调用LLM、执行工具）定义为图中的一个**节点 (Node)**，并用**边 (Edge)** 来定义节点之间的跳转逻辑。这种设计天然支持**循环 (Cycles)**，使得实现如 Reflection 这样的迭代、修正、自我反思的复杂工作流变得异常简单和直观



出现的问题：

这里的理解与查询节点逻辑是，先获取用户的最后一次回答，然后分析用户需求并生成关键搜索词。

此处采用的操作是无条件取最后一条消息，因为没有判断内容来自哪个对象，所以可能会取到`AIMessage`，

```python
def understand_query_node(state: SearchState) -> dict:
    """步骤1：理解用户查询并生成搜索关键词"""
    user_message = state["messages"][-1].content
    # .content 针对 对象取属性("content")的内容 因为OpenAI官方定义的标准就是这个，详见client.py
    
    understand_prompt = f"""分析用户的查询："{user_message}"
请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content
    ...
```

改正：
```python
user_message = ""
# 获取最后一个消息对话
# 倒序遍历,找到最近的一条用户消息
for msg in reversed(state["messages"]):
	if isinstance(msg, HumanMessage):
	​	user_message = msg.content
		break

```

还有一个潜在的缺陷：当前代码只取最近一句用户输入，不能完整理解上下文依赖

### 构造Agent架构 By myself

项目目录
```
hello-agents/
├── hello_agents/
│   │
│   ├── core/                     # 核心框架层
│   │   ├── agent.py              # Agent基类
│   │   ├── llm.py                # HelloAgentsLLM统一接口
│   │   ├── message.py            # 消息系统
│   │   ├── config.py             # 配置管理
│   │   └── exceptions.py         # 异常体系
│   │
│   ├── agents/                   # Agent实现层
│   │   ├── simple_agent.py       # SimpleAgent实现
│   │   ├── react_agent.py        # ReActAgent实现
│   │   ├── reflection_agent.py   # ReflectionAgent实现
│   │   └── plan_solve_agent.py   # PlanAndSolveAgent实现
│   │
│   ├── tools/                    # 工具系统层
│   │   ├── base.py               # 工具基类
│   │   ├── registry.py           # 工具注册机制
│   │   ├── chain.py              # 工具链管理系统
│   │   ├── async_executor.py     # 异步工具执行器
│   │   └── builtin/              # 内置工具集
│   │       ├── calculator.py     # 计算工具
│   │       └── search.py         # 搜索工具
└──
```

