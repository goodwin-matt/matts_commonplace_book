# How to build

1. Add/edit content
2. Run `jb build matts_commonplace_book` in the parent directory.
   This will build static files in the `_build` folder. These files can then be previewed.
3. Once the previews are satisfactory, and we want to push to a live website run `ghp-import -n -p -f _build/html`.
   You may have to delete the `_build` folder first 
4. Navigate to https://goodwin-matt.github.io/matts_commonplace_book/intro.html
