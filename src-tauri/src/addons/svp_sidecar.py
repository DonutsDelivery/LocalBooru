"""Start SVP on a private loopback socket owned continuously by this child."""
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    # A binary handshake keeps Windows text-mode CRLF translation out of the wire format.
    sys.stdout.buffer.write(f"DMC_SVP_PORT={listener.getsockname()[1]}\n".encode("ascii"))
    sys.stdout.buffer.flush()
    import uvicorn

    uvicorn.Server(uvicorn.Config("app:app", host="127.0.0.1", port=0)).run(
        sockets=[listener]
    )
