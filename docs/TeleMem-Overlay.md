<div align="center">
  <h1>TeleMem Overlay Development Guide</h1>
  <p>
      <a href="TeleMem-Overlay.md">English</a> | <a href="TeleMem-Overlay-ZH.md">简体中文</a>
  </p>
</div>

> **Goal**: Extend upstream repositories (e.g., [mem0](https://github.com/mem0ai/mem0)) with custom features—such as modifying the behavior of `Memory.add()`—**without directly altering the original codebase**, while retaining the ability to seamlessly sync future upstream updates.

---

## 🧩 What Is Overlay Mode?

Overlay mode is a lightweight approach to extending upstream repositories. It allows you to:

- Keep the original upstream code **clean and updatable**;
- Add local features or fixes, and save those changes as **patch files**;
- **One-click sync** with upstream updates and automatically reapply your patches.

You **do not need** to modify the upstream Git history or maintain a long-term fork. 

---

## ⚙️ Initialize the Upstream Repository

On first setup, run:

```bash
export UPSTREAM_REPO="https://github.com/mem0ai/mem0.git"
export UPSTREAM_REF="main"
bash scripts/init_upstream.sh
```

This will:

- Clone the upstream repository;
- Place it under `vendor/mem0/`;
- Automatically create an initial commit.

---

## 🧱 Create Your First Patch

Suppose you want to modify the upstream `Memory.add()` function by adding a custom logic hook.

### 1️⃣ Modify the Upstream Code

Edit the file:

```bash
vim vendor/mem0/mem0/memory/main.py
```

Add your custom logic:

```python
def add(self, *args, **kwargs):
    print("[TeleMem] Hook before add()")  # custom extension
    return super().add(*args, **kwargs)
```

---

### 2️⃣ Record the Patch

Run:

```bash
bash scripts/record_patch.sh add-memory-hook
```

This script will:

- Extract changes made within `vendor/mem0`;
- Save them as `overlay/patches/add-memory-hook.patch`;
- Log the patch in `PATCHES.md`;
- Remind you to restore the upstream to a clean state.

---

### 3️⃣ Restore Upstream Code

Since `vendor/mem0` is a **read-only zone** (meant to hold pristine upstream code), restore it:

```bash
git checkout vendor/mem0
```

---

### 4️⃣ Apply the Patch

When you need to run or build, reapply patches:

```bash
bash scripts/apply_patches.sh
```

This applies all `.patch` files in order to `vendor/mem0/`.

---

## 🔁 Sync Upstream Updates

When the upstream (`mem0`) releases a new version, simply run:

```bash
bash scripts/update_upstream.sh
```

This automatically:

- Pulls the latest upstream version;
- Reapplies all patches via `apply_patches.sh`;
- Checks for conflicts (manual resolution required if any).

---

## 🧰 Patch Management

All patches are stored in:

```
overlay/patches/*.patch
```

Each patch corresponds to a logical change, for example:

- `add-memory-hook.patch`
- `fix-config-loading.patch`
- `extend-llm-registry.patch`

Maintain descriptions in `PATCHES.md`, e.g.:

```markdown
# PATCHES.md

- add-memory-hook (2025-10-23): Add TeleMem hook before Memory.add()
- extend-llm-registry (2025-10-24): Add new provider registry hook for reranker
```

---

## 🧩 Script Reference

| Script               | Purpose                                      | Usage                                 |
| -------------------- | -------------------------------------------- | ------------------------------------- |
| `init_upstream.sh`   | Import upstream code for the first time      | `bash scripts/init_upstream.sh`       |
| `update_upstream.sh` | Pull upstream updates & reapply patches      | `bash scripts/update_upstream.sh`     |
| `record_patch.sh`    | Save current upstream modifications as patch | `bash scripts/record_patch.sh <name>` |
| `apply_patches.sh`   | Reapply all patches                          | `bash scripts/apply_patches.sh`       |

---

## 🧩 Best Practices

- Keep each logical change in a **single, small patch** (aim for ≤10 patches total).
- **Never commit** directly modified files inside `vendor/`.
- Always reapply patches before testing or building:

```bash
bash scripts/apply_patches.sh
```

- Always sync upstream using:

  ```bash
  bash scripts/update_upstream.sh
  ```

- If upstream changes cause conflicts, consider **contributing a hook point** upstream via a PR instead of patching deeply.

---

## 🧩 Workflow Summary

```bash
# First-time initialization
bash scripts/init_upstream.sh

# Modify upstream (vendor/mem0)
vim vendor/mem0/mem0/memory/main.py

# Record the change
bash scripts/record_patch.sh add-memory-hook

# Restore upstream to clean state
git checkout vendor/mem0

# Reapply patches before running
bash scripts/apply_patches.sh

# Sync when upstream updates
bash scripts/update_upstream.sh
```

---

## ✅ Benefits Summary

| Feature          | Description                                          |
| ---------------- | ---------------------------------------------------- |
| ✅ No Fork Needed | Avoid maintaining your own `mem0` branch             |
| ✅ Easy Sync      | `git subtree pull` + `apply_patches`                 |
| ✅ Traceable      | All changes are isolated in `.patch` files           |
| ✅ Automated      | Scripts rebuild any version reliably                 |
| ✅ Auditable      | `PATCHES.md` documents all modifications & rationale |

---

## 🔐 Important Notes

- **Never directly commit** changes inside `vendor/mem0`.
- If a patch fails (due to conflicts), use `git apply --reject` and resolve manually.
- Always keep `PATCHES.md` updated to support team collaboration.
- It’s recommended to commit both `overlay/patches/` and `scripts/` into version control.
