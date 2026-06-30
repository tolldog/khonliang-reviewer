The following are examples of findings you must NOT produce. Each is a
real false-positive pattern observed from local-model reviews. They are
shown so you can recognize and avoid them — do not emit anything shaped
like these.

### Anti-example 1 — echo-as-finding (restating the change is not a critique)

```json
{"severity": "concern", "title": "Updates the docstring",
 "body": "This change updates the function's docstring to describe the new
 parameter and its default value."}
```

WHY THIS IS WRONG: the "finding" merely restates what the change does. It
asserts nothing the code gets wrong — no bug, risk, or omission. Describing
or summarizing a diff is never a finding. If the change is sound, emit
nothing for it.

### Anti-example 2 — inverted claim (asserting the opposite of the change)

```json
{"severity": "concern", "title": "Code Repetition",
 "body": "The literal value is duplicated and should be extracted into a
 single constant."}
```

WHY THIS IS WRONG: this fires on a diff that CONSOLIDATES two duplicate
literals into one constant — i.e. the change already does exactly what the
finding asks for. The finding asserts the opposite of what the diff does.
Always confirm the change still exhibits the problem before flagging it.

### Anti-example 3 — style preference raised to `concern`

```json
{"severity": "concern", "title": "Prefer f-strings",
 "body": "Consider using an f-string instead of str.format() here."}
```

WHY THIS IS WRONG: a subjective style/formatting preference is not a
blocking defect. If worth mentioning at all, it is a `comment` or `nit` —
never a `concern`. Reserve `concern` for something that should stop the
change (a real bug, regression, or security problem).
