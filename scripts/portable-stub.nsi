; LocalBooru Portable Stub Launcher
; Tiny exe (~50KB) that just launches LocalBooru\LocalBooru.exe relative to itself.
; After first extraction, the big self-extractor replaces itself with this stub
; so subsequent launches are instant.

Name "LocalBooru Portable"
OutFile "..\dist\portable-stub.exe"
Icon "..\assets\icon.ico"
RequestExecutionLevel user
SilentInstall silent

Section
    ; $EXEDIR = directory where this exe lives
    ; LocalBooru.exe is in LocalBooru\ subfolder next to this exe
    StrCpy $0 "$EXEDIR\LocalBooru\LocalBooru.exe"

    IfFileExists $0 launch error

launch:
    Exec '"$0"'
    Goto done

error:
    MessageBox MB_OK|MB_ICONERROR "LocalBooru application files not found.$\r$\n$\r$\nExpected: $0$\r$\n$\r$\nPlease re-download LocalBooru-Portable.exe."

done:
SectionEnd
