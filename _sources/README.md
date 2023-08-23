# How to build

1. Add/edit contenct
2. Run `jb build the_data_scientists_commonplace_book`. This will build static files in the `_build`
folder. These files can then be previewed.
3. Once the previews are satisfactory, and we want to push to a live website run `ghp-import -n -p -f _build/html`
