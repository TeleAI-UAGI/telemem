<div align="center">
  <h1>TeleMem Overlay 开发说明</h1>
  <p>
      <a href="TeleMem-Overlay.md">English</a> | <a href="TeleMem-Overlay-ZH.md">简体中文</a>
  </p>
</div>

> **目标**：在不直接修改上游仓库（如 [mem0](https://github.com/mem0ai/mem0)）的情况下，为其添加扩展功能（如自定义 `Memory.add()` 行为），并保持可随时同步上游更新。

---

## 🧩 Overlay 模式是什么？

Overlay 模式是一种轻量化的上游扩展方式，它允许你：

- **保留上游仓库的原始代码**（干净、可更新）；
- **在本地添加功能或修复**，并将改动保存为补丁文件；
- **一键同步上游更新并重新打补丁**。

你不需要改动上游的 git 历史，也不必维护一个长期分叉（fork）。  

---

## ⚙️ 初始化上游仓库

第一次初始化时，执行：

```bash
export UPSTREAM_REPO="https://github.com/mem0ai/mem0.git"
export UPSTREAM_REF="main"
bash scripts/init_upstream.sh
```

这会：

- 下载上游仓库；
- 放入 `vendor/mem0/`；
- 自动创建初始提交。

---

## 🧱 创建你的第一个补丁

假设我们要修改上游的 `Memory.add()` 函数，加一行自定义逻辑。

### 1️⃣ 修改上游代码

编辑文件：

```bash
vim vendor/mem0/mem0/memory/main.py
```

添加自定义逻辑：

```python
def add(self, *args, **kwargs):
    print("[TeleMem] Hook before add()")  # custom extension
    return super().add(*args, **kwargs)
```

---

### 2️⃣ 记录补丁

执行命令：

```bash
bash scripts/record_patch.sh add-memory-hook
```

脚本会：

- 提取 `vendor/mem0` 的改动；
- 保存为 `overlay/patches/add-memory-hook.patch`；
- 在 `PATCHES.md` 里记录说明；
- 提醒你恢复上游为干净状态。

---

### 3️⃣ 恢复上游代码

因为 `vendor/mem0` 是只读区域（存放干净的上游代码），  
我们恢复原样以保持可同步性：

```bash
git checkout vendor/mem0
```

---

### 4️⃣ 应用补丁

当你需要运行或构建时，重新打上补丁：

```bash
bash scripts/apply_patches.sh
```

这会将所有 `.patch` 文件按顺序自动应用到 `vendor/mem0/`。

---

## 🔁 同步上游更新

当上游（mem0）发布新版本时，只需执行：

```bash
bash scripts/update_upstream.sh
```

它会自动：

1. 拉取上游最新版本；
2. 执行 `apply_patches.sh`；
3. 检查是否存在冲突（若有需手动调整 patch）。

---

## 🧰 管理补丁

所有补丁都位于：

```
overlay/patches/*.patch
```

每个补丁对应一个逻辑修改点，例如：

- `add-memory-hook.patch`  
- `fix-config-loading.patch`  
- `extend-llm-registry.patch`  

你可以在 `PATCHES.md` 中维护说明，例如：

```markdown
# PATCHES.md

- add-memory-hook (2025-10-23): Add TeleMem hook before Memory.add()
- extend-llm-registry (2025-10-24): Add new provider registry hook for reranker
```

---

## 🧩 脚本说明

| 脚本 | 作用 | 用法 |
|------|------|------|
| `init_upstream.sh` | 第一次引入上游代码 | `bash scripts/init_upstream.sh` |
| `update_upstream.sh` | 从上游更新并自动重打补丁 | `bash scripts/update_upstream.sh` |
| `record_patch.sh` | 将当前对上游的改动保存为补丁 | `bash scripts/record_patch.sh <patch-name>` |
| `apply_patches.sh` | 重新应用所有补丁 | `bash scripts/apply_patches.sh` |

---

## 🧩 使用建议

- 每个逻辑修改只生成一个补丁，**保持粒度小（≤10 个补丁）**
- **不要直接提交修改后的 vendor/** 内容；
- 任何时候想测试或构建，可执行：
  ```bash
  bash scripts/apply_patches.sh
  ```
- 同步上游时总是通过：
  ```bash
  bash scripts/update_upstream.sh
  ```
- 如果上游有冲突，优先考虑在上游仓库提 PR 增加「hook」点。

---

## 🧩 工作流程一览

```bash
# 第一次初始化
bash scripts/init_upstream.sh

# 修改上游（vendor/mem0）
vim vendor/mem0/mem0/memory/main.py

# 记录改动
bash scripts/record_patch.sh add-memory-hook

# 恢复上游
git checkout vendor/mem0

# 构建或运行前重新打补丁
bash scripts/apply_patches.sh

# 当上游更新
bash scripts/update_upstream.sh
```

---

## ✅ 优点总结

| 特性 | 说明 |
|------|------|
| ✅ 无需 fork | 不维护自己的 mem0 分支 |
| ✅ 易于同步 | `subtree pull` + `apply_patches` |
| ✅ 改动可追踪 | 所有修改都在补丁文件中 |
| ✅ 自动化脚本 | 轻松重建任何版本 |
| ✅ 可审计 | `PATCHES.md` 记录了所有修改来源和理由 |

---

## 🔐 注意事项

- `vendor/mem0` 中的文件永远不应直接 commit；
- 如果 patch 失败（冲突），请使用 `git apply --reject` 手动修复；
- 始终保持 `PATCHES.md` 更新，方便团队协作；
- 推荐将 `overlay/patches` 和 `scripts` 一并提交入版本控制。
