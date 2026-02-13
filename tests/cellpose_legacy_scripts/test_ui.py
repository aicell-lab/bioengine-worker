import http.server
import os
import socketserver
import webbrowser

PORT = 8080
DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)


def run():
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving UI at http://localhost:{PORT}")
        print(f"Directory: {DIRECTORY}")
        print("Press Ctrl+C to stop.")

        # Try to open in browser
        try:
            webbrowser.open(f"http://localhost:{PORT}/index.html")
        except:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")
            httpd.shutdown()


if __name__ == "__main__":
    run()
