# Using Superpowers

## 用途

在其他工程或新会话里，让 AI 先加载 Superpowers 工程开发流程，之后按规范执行需求澄清、设计、计划、实现和验证。

## 推荐输入

把下面这段作为新会话第一句话发给 AI：

```text
/using-superpowers 启动，规范工程开发全流程。请先读取：
cat ~/.ai-superpowers/all.md
然后按 using-superpowers 规则工作。
```

## 简短输入

```text
/using-superpowers
cat ~/.ai-superpowers/all.md
```

## 核心技能速查

不需要把这些技能文件复制到每个工程。每个工程只保留这个索引，完整规则以
`~/.ai-superpowers/all.md` 为准。

1. `brainstorm`：需求深挖，补齐目标、边界条件和成功标准。
2. `plan`：任务拆分，明确文件、接口、步骤和验证命令。
3. `tdd`：测试驱动；能写测试的业务逻辑先写测试。
4. `structure`：统一项目目录、模块边界和文件职责。
5. `style`：代码风格、命名、注释和一致性规范。
6. `debug`：系统化排障，定位根因，不靠猜测打补丁。
7. `refactor`：安全重构，保持行为不变，逐步验证。
8. `review`：自检代码、文档、风险和规范符合度。
9. `doc`：生成接口、运行、部署和使用文档。
10. `release`：版本打包、发布前检查和交付校验。

## 日常工作流

```text
需求不清 -> brainstorm 澄清
目标明确 -> plan 拆任务
可测试逻辑 -> tdd 先写用例
实现代码 -> 按项目 structure/style 落地
遇到问题 -> debug 追根因
改动完成 -> review 自检
需要交付 -> doc/release 补齐文档和发布校验
```

## 注意

- 这是发给 AI 的指令，不是在 shell 里直接运行。
- 如果某个工程希望默认启用，把推荐输入写进该工程的 `AGENTS.md`、`CLAUDE.md` 或项目 README。
- 后续让 AI 改代码前，应先确认目标、方案和验证命令。
