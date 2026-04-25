---
name: refactoring-patterns
description: "Safe refactoring techniques — extract, inline, rename, move, simplify conditionals, and decompose functions. Use when restructuring code without changing behavior. TRIGGER when: refactor, extract method, rename, inline, simplify, decompose, clean up, code smell. DO NOT TRIGGER when: adding features, fixing bugs, writing tests, documentation."
allowed-tools: [Read, Grep, Glob]
---

# Refactoring Patterns

Safe techniques for restructuring code without changing its behavior. Every refactoring follows: test green -> refactor -> test green.

## Golden Rule

**Never refactor and change behavior in the same commit.** Refactoring = same behavior, different structure. Feature work = different behavior. Mixing them makes bugs untraceable.

## Pre-Refactoring Checklist

Before any refactoring:
- [ ] Tests exist and pass for the code being refactored
- [ ] You can describe what the code does WITHOUT reading it line by line
- [ ] The refactoring has a clear motivation (not "it could be cleaner")
- [ ] The scope is bounded — you know exactly which files/functions change

## Core Refactorings

### 1. Extract Function

**When**: A code block does one identifiable thing, or you need to add a comment explaining what a block does.

```python
# BEFORE
def process_order(order):
    # Validate order
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Negative total")
    if not order.customer_id:
        raise ValueError("Missing customer")

    # Apply discounts
    total = order.total
    if order.is_member:
        total *= 0.9
    if len(order.items) > 10:
        total *= 0.95

    return total

# AFTER
def process_order(order):
    validate_order(order)
    return apply_discounts(order)

def validate_order(order):
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Negative total")
    if not order.customer_id:
        raise ValueError("Missing customer")

def apply_discounts(order):
    total = order.total
    if order.is_member:
        total *= 0.9
    if len(order.items) > 10:
        total *= 0.95
    return total
```

**Verification**: Run tests. Output must be identical.

### 2. Inline Function

**When**: A function's body is as clear as its name, or it's only called once and adds indirection without value.

```python
# BEFORE
def is_valid_age(age):
    return age >= 0

def process(age):
    if is_valid_age(age):
        ...

# AFTER (if is_valid_age is only used once and obvious)
def process(age):
    if age >= 0:
        ...
```

**When NOT to inline**: If the function is called from multiple places, or if the name adds clarity that the body doesn't.

### 3. Rename

**When**: A name doesn't describe what the thing does, or uses abbreviations/jargon.

```python
# BEFORE
def proc(d, f=True):
    ...
x = get_data()
tmp = transform(x)

# AFTER
def process_invoice(invoice_data, *, validate=True):
    ...
raw_invoices = fetch_invoices()
normalized_invoices = normalize(raw_invoices)
```

**Rules**:
- Search for ALL usages before renaming (Grep across entire codebase)
- Update imports, tests, documentation, and config files
- If it's a public API, this is a BREAKING CHANGE — requires version bump

### 4. Move

**When**: A function/class is in the wrong module — it's more closely related to another module's concerns.

**Process**:
1. Grep for all imports of the function/class
2. Move to new location
3. Update all import statements
4. Add re-export from old location if it's a public API (temporary, with deprecation warning)
5. Run tests

### 5. Simplify Conditionals

**When**: Nested if/else chains, complex boolean expressions, or repeated condition checks.

```python
# BEFORE: Nested guards
def get_price(product, user):
    if product is not None:
        if product.is_available:
            if user is not None:
                if user.is_member:
                    return product.price * 0.9
                else:
                    return product.price
            else:
                return product.price
        else:
            return None
    else:
        return None

# AFTER: Early returns (guard clauses)
def get_price(product, user):
    if product is None or not product.is_available:
        return None

    if user is not None and user.is_member:
        return product.price * 0.9

    return product.price
```

### 6. Decompose Large Functions

**When**: A function is longer than ~30 lines or has multiple levels of abstraction.

**Process**:
1. Identify logical sections (often marked by comments or blank lines)
2. Extract each section into a named function
3. The parent function should read like a table of contents
4. Each extracted function should work at one level of abstraction

### 7. Replace Magic Values

```python
# BEFORE
if response.status_code == 429:
    time.sleep(60)

# AFTER
RATE_LIMIT_STATUS = 429
DEFAULT_RETRY_DELAY_SECONDS = 60

if response.status_code == RATE_LIMIT_STATUS:
    time.sleep(DEFAULT_RETRY_DELAY_SECONDS)
```

## Code Smells That Signal Refactoring

| Smell | Refactoring |
|-------|-------------|
| Long function (>30 lines) | Extract Function, Decompose |
| Deeply nested conditionals | Guard Clauses, Extract Function |
| Duplicated code blocks | Extract Function, parameterize |
| Feature envy (method uses another class's data more than its own) | Move Method |
| Long parameter list (>4 params) | Introduce Parameter Object |
| Comments explaining "what" (not "why") | Rename, Extract Function |
| Boolean parameters | Split into two functions |
| Dead code | Delete it |

## Refactoring Safety

### Test-First Verification
```bash
# 1. Confirm tests pass BEFORE refactoring
python -m pytest tests/ -x --tb=short

# 2. Make the refactoring change

# 3. Confirm tests STILL pass
python -m pytest tests/ -x --tb=short

# 4. Verify no behavioral change
git diff  # Review: structure changes only, no logic changes
```

### When to Stop
- FORBIDDEN: "While I'm here, I'll also..." — scope creep
- FORBIDDEN: Refactoring code you don't have tests for
- FORBIDDEN: Refactoring and adding features in the same commit
- FORBIDDEN: Premature abstraction — three similar blocks is NOT a pattern until you see the fourth

### Scale Guidelines
| Scope | Approach |
|-------|----------|
| Single function | Refactor inline, same commit |
| Single file | Refactor in dedicated commit |
| Multiple files | Plan first, dedicated branch |
| Module/package boundary | ADR required, phased approach |
