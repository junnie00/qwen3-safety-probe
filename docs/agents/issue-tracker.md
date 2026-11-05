# 问题跟踪：GitHub

本仓库的需求、问题与规格保存在 `junnie00/qwen3-safety-probe` 的 GitHub Issues，使用 `gh` CLI 读写，不以本地 `.scratch/` 文件替代正式工单。

## 仓库定位与写入安全

- 在仓库根目录运行 `gh`，由 `origin` 推断目标仓库；仓库地址为 `https://github.com/junnie00/qwen3-safety-probe`。
- 每次会话首次创建 issue 前，执行 `gh repo view --json nameWithOwner`，确认目标为本仓库。若目标变化、指向上游或存在歧义，先澄清，不向其他仓库写入。
- 不向技能来源仓库 `mattpocock/skills` 提交本项目问题。
- 工单、评论和规格正文使用中文；代码标识、标签和专有名称按原文保留。

## 常用操作

- 创建：`gh issue create --title "标题" --body "正文"`；多行正文可用 Bash heredoc 通过 `--body-file -` 输入。
- 读取：`gh issue view <编号> --json number,title,body,labels,comments`。
- 列表：`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`；按需增加 `--label`、`--state`，处理完整列表时注意分页或 `--limit`。
- 评论：`gh issue comment <编号> --body "说明"`。
- 标签：`gh issue edit <编号> --add-label "标签"` 或 `--remove-label "标签"`；标签语义见 `docs/agents/triage-labels.md`。
- 关闭：先评论说明，再 `gh issue close <编号>`；工单已关闭时仍可另发说明评论。

## 父子工单与阻塞关系

优先使用 GitHub 原生关系，不把普通正文链接误当作已经建立原生关系。

- gh 2.94 及以上创建子工单：`gh issue create --parent <父编号> --title "标题" --body "正文"`；关联已有子工单：`gh issue edit <父编号> --add-sub-issue <子编号>`。
- gh 2.94 及以上增加阻塞：`gh issue edit <被阻塞编号> --add-blocked-by <阻塞者编号>`；创建时可用 `--blocked-by <编号>,<编号>`。
- 老版本可用 `gh api --method POST repos/{owner}/{repo}/issues/<父编号>/sub_issues -F sub_issue_id=<子工单数据库ID>` 建立父子关系，用 `gh api --method POST repos/{owner}/{repo}/issues/<被阻塞编号>/dependencies/blocked_by -F issue_id=<阻塞者数据库ID>` 建立依赖。
- 数据库 ID 用 `gh api repos/{owner}/{repo}/issues/<编号> --jq .id` 查询；不是 `#编号`，也不是 `node_id`。
- 查询未关闭阻塞数：`gh api repos/{owner}/{repo}/issues/<编号> --jq .issue_dependencies_summary.blocked_by`；确认关系可用后，0 表示不存在未关闭的阻塞。
- 原生父子关系不可用时，在子工单顶部记录 `Part of #<父编号>`，并在父工单中维护任务列表；依赖不可用时，在被阻塞工单顶部记录 `Blocked by: #<编号>, #<编号>`。所有阻塞者关闭后才视为解除阻塞。

## PR 作为需求入口

**PRs as a request surface: no.**

默认不把外部 PR 纳入需求分类队列。若日后改为 yes，使用 `gh pr view <编号>`、`gh pr view <编号> --comments` 和 `gh pr diff <编号>` 读取正文、评论与 diff，通过 `gh pr comment`、`gh pr edit --add-label` / `--remove-label` 和 `gh pr close` 操作。

启用后可用以下查询识别外部贡献者的开放 PR：

`gh api --paginate 'repos/{owner}/{repo}/pulls?state=open' --jq '.[] | select(.author_association | IN("OWNER","MEMBER","COLLABORATOR") | not) | {number, title, author: .user.login, author_association, labels: [.labels[].name]}'`

GitHub Issues 与 PR 共用编号空间；无法确定 `#编号` 的类型时，先用 `gh pr view <编号>` 确认，非 PR 再读取 issue。

## 技能指令的含义

- “发布到 issue tracker”：创建本仓库的 GitHub Issue。
- “获取相关 ticket”：读取对应 Issue，包括正文、标签和相关评论。

## Wayfinder 操作

- 探索地图是标注 `wayfinder:map` 的单个 Issue，正文保存笔记、已有决定及尚未解决的问题。
- 地图下的工作项使用原生子工单；不可用时使用父工单任务列表与子工单 `Part of #<父编号>`。类型标签为 `wayfinder:research`、`wayfinder:prototype`、`wayfinder:grilling` 或 `wayfinder:task`。这些不是本次 setup 要创建的 triage 标签，按需使用。
- 可领取工作项必须属于该地图、尚未关闭、没有未关闭阻塞且尚未分配负责人；按地图顺序选择。
- 领取：`gh issue edit <编号> --add-assignee @me`，作为该工作会话的首次写操作。
- 解决：评论答案后关闭工作项，再把简要结论和工单链接追加到地图的已有决定中。
