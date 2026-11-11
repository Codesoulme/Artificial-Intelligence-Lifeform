# Named address binding after structural replacement

The finite name-to-address projection was retrained using features generated
after the SPORA replacements at layers 6 and 12. It recovered both address
sites for all 32 supervised training questions. On 16 held questions for known
names, all 8 name bindings were correct; all 8 unknown names were rejected.

The artifact records the candidate checkpoint digest and structural feature
variant. It remains separate from the live runtime because the test uses
supervised names and synthetic question templates.
