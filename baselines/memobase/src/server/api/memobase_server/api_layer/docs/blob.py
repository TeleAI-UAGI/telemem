from .basic_docs import add_api_code_docs, py_code, js_code, go_code

# Get user blobs by type
add_api_code_docs(
    "GET",
    "/users/blobs/{user_id}/{blob_type}",
    py_code(
        """
from memobase import MemoBaseClient
from memobase.core.blob import BlobType

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

user = client.get_user('user_id')
blobs = user.get_all(BlobType.chat)
"""
    ),
    js_code(
        """
import { MemoBaseClient, BlobType } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);

const user = client.getUser('user_id');
const blobs = await user.getAll(BlobType.Enum.chat);
"""
    ),
    go_code(
        """
import (
    "fmt"
    "log"

    "github.com/memodb-io/memobase/src/client/memobase-go/blob"
    "github.com/memodb-io/memobase/src/client/memobase-go/core"
)

func main() {
    projectURL := "YOUR_PROJECT_URL"
    apiKey := "YOUR_API_KEY"
    // Initialize the client
    client, err := core.NewMemoBaseClient(
        projectURL,
        apiKey,
    )
    if err != nil {
        log.Fatalf("Failed to create client: %v", err)
    }

    // Get a user
    userID := "EXISTING_USER_ID" // Replace with an actual user ID
    user, err := client.GetUser(userID, false)
    if err != nil {
        log.Fatalf("Failed to get user: %v", err)
    }

    // Get all chat blobs
    blobIDs, err := user.GetAll(blob.ChatType, 0, 10)
    if err != nil {
        log.Fatalf("Failed to get blobs: %v", err)
    }

    fmt.Printf("Found %d chat blobs\n", len(blobIDs))
}
"""
    ),
)

# Insert blob
add_api_code_docs(
    "POST",
    "/blobs/insert/{user_id}",
    py_code(
        """
from memobase import MemoBaseClient
from memobase.core.blob import ChatBlob

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

b = ChatBlob(messages=[
    {
        "role": "user",
        "content": "Hi, I'm here again"
    },
    {
        "role": "assistant",
        "content": "Hi, Gus! How can I help you?"
    }
])
u = client.get_user(uid)
bid = u.insert(b)
"""
    ),
    js_code(
        """
import { MemoBaseClient, Blob, BlobType } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);
const user = await client.getUser(userId);

const blobId = await user.insert(Blob.parse({
  type: BlobType.Enum.chat,
  messages: [
    {
      role: 'user',
      content: 'Hi, I\'m here again'
    },
    {
      role: 'assistant',
      content: 'Hi, Gus! How can I help you?'
    }
  ]
}));
"""
    ),
    go_code(
        """
import (
    "fmt"
    "log"

    "github.com/memodb-io/memobase/src/client/memobase-go/blob"
    "github.com/memodb-io/memobase/src/client/memobase-go/core"
)

func main() {
    projectURL := "YOUR_PROJECT_URL"
    apiKey := "YOUR_API_KEY"
    // Initialize the client
    client, err := core.NewMemoBaseClient(
        projectURL,
        apiKey,
    )
    if err != nil {
        log.Fatalf("Failed to create client: %v", err)
    }

    // Get a user
    userID := "EXISTING_USER_ID" // Replace with an actual user ID
    user, err := client.GetUser(userID, false)
    if err != nil {
        log.Fatalf("Failed to get user: %v", err)
    }

    // Create a chat blob
    chatBlob := &blob.ChatBlob{
        BaseBlob: blob.BaseBlob{
            Type: blob.ChatType,
        },
        Messages: []blob.OpenAICompatibleMessage{
            {
                Role:    "user",
                Content: "Hello, I am Jinjia!",
            },
            {
                Role:    "assistant",
                Content: "Hi there! How can I help you today?",
            },
        },
    }

    // Insert the blob
    blobID, err := user.Insert(chatBlob, false)
    if err != nil {
        log.Fatalf("Failed to insert blob: %v", err)
    }
    fmt.Printf("Successfully inserted blob with ID: %s\n", blobID)
}
"""
    ),
)

# Get blob
add_api_code_docs(
    "GET",
    "/blobs/{user_id}/{blob_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

u = client.get_user(uid)
b = u.get(bid)
"""
    ),
    js_code(
        """
import { MemoBaseClient } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);
const user = await client.getUser(userId);

const blob = await user.get(blobId);
"""
    ),
    go_code(
        """
import (
    "fmt"
    "log"

    "github.com/memodb-io/memobase/src/client/memobase-go/blob"
    "github.com/memodb-io/memobase/src/client/memobase-go/core"
)

func main() {
    projectURL := "YOUR_PROJECT_URL"
    apiKey := "YOUR_API_KEY"
    // Initialize the client
    client, err := core.NewMemoBaseClient(
        projectURL,
        apiKey,
    )
    if err != nil {
        log.Fatalf("Failed to create client: %v", err)
    }

    // Get a user
    userID := "EXISTING_USER_ID" // Replace with an actual user ID
    user, err := client.GetUser(userID, false)
    if err != nil {
        log.Fatalf("Failed to get user: %v", err)
    }

    // Get a blob
    blobID := "EXISTING_BLOB_ID" // Replace with an actual blob ID
    retrievedBlob, err := user.Get(blobID)
    if err != nil {
        log.Fatalf("Failed to get blob: %v", err)
    }

    // Type assert to use as ChatBlob
    if chatBlob, ok := retrievedBlob.(*blob.ChatBlob); ok {
        fmt.Printf("Retrieved message: %s\n", chatBlob.Messages[0].Content)
    }
}
"""
    ),
)

# Delete blob
add_api_code_docs(
    "DELETE",
    "/blobs/{user_id}/{blob_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

u = client.get_user(uid)
u.delete(bid)
"""
    ),
    js_code(
        """
import { MemoBaseClient } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);
const user = await client.getUser(userId);

await user.delete(blobId);
"""
    ),
    go_code(
        """
import (
    "fmt"
    "log"

    "github.com/memodb-io/memobase/src/client/memobase-go/core"
)

func main() {
    projectURL := "YOUR_PROJECT_URL"
    apiKey := "YOUR_API_KEY"
    // Initialize the client
    client, err := core.NewMemoBaseClient(
        projectURL,
        apiKey,
    )
    if err != nil {
        log.Fatalf("Failed to create client: %v", err)
    }

    // Get a user
    userID := "EXISTING_USER_ID" // Replace with an actual user ID
    user, err := client.GetUser(userID, false)
    if err != nil {
        log.Fatalf("Failed to get user: %v", err)
    }

    // Delete a blob
    blobID := "EXISTING_BLOB_ID" // Replace with an actual blob ID
    err = user.Delete(blobID)
    if err != nil {
        log.Fatalf("Failed to delete blob: %v", err)
    }
    fmt.Printf("Successfully deleted blob with ID: %s\n", blobID)
}
"""
    ),
)

# Buffer operations
add_api_code_docs(
    "POST",
    "/users/buffer/{user_id}/{buffer_type}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

u = client.get_user(uid)
u.flush()
u.flush(sync=True) # wait for the buffer to be processed
"""
    ),
    js_code(
        """
import { MemoBaseClient, BlobType } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);
const user = await client.getUser(userId);

await user.flush(BlobType.Enum.chat);
"""
    ),
    go_code(
        """
import (
    "fmt"
    "log"

    "github.com/memodb-io/memobase/src/client/memobase-go/blob"
    "github.com/memodb-io/memobase/src/client/memobase-go/core"
)

func main() {
    projectURL := "YOUR_PROJECT_URL"
    apiKey := "YOUR_API_KEY"
    // Initialize the client
    client, err := core.NewMemoBaseClient(
        projectURL,
        apiKey,
    )
    if err != nil {
        log.Fatalf("Failed to create client: %v", err)
    }

    // Get a user
    userID := "EXISTING_USER_ID" // Replace with an actual user ID
    user, err := client.GetUser(userID, false)
    if err != nil {
        log.Fatalf("Failed to get user: %v", err)
    }

    // Flush the buffer
    err = user.Flush(blob.ChatType, false)
    if err != nil {
        log.Fatalf("Failed to flush buffer: %v", err)
    }
    fmt.Println("Successfully flushed buffer")
}
"""
    ),
)

# Get buffer capacity
add_api_code_docs(
    "GET",
    "/users/buffer/capacity/{user_id}/{buffer_type}",
    py_code(
        """
from memobase import MemoBaseClient
from memobase.core.blob import BlobType

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

user = client.get_user('user_id')
blobs = user.buffer(BlobType.chat)
"""
    ),
    go_code(
        """
import (
    "fmt"
    "log"

    "github.com/memodb-io/memobase/src/client/memobase-go/blob"
    "github.com/memodb-io/memobase/src/client/memobase-go/core"
)

func main() {
    projectURL := "YOUR_PROJECT_URL"
    apiKey := "YOUR_API_KEY"
    // Initialize the client
    client, err := core.NewMemoBaseClient(
        projectURL,
        apiKey,
    )
    if err != nil {
        log.Fatalf("Failed to create client: %v", err)
    }

    // Get a user
    userID := "EXISTING_USER_ID" // Replace with an actual user ID
    user, err := client.GetUser(userID, false)
    if err != nil {
        log.Fatalf("Failed to get user: %v", err)
    }

    // Get buffer capacity
    blobIDs, err := user.Buffer(blob.ChatType, "processing")
    if err != nil {
        log.Fatalf("Failed to get buffer capacity: %v", err)
    }
    fmt.Printf("Found %d blobs in buffer\n", len(blobIDs))
}
"""
    ),
)
