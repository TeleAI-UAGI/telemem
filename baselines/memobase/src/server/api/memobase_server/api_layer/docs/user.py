from .basic_docs import add_api_code_docs, py_code, js_code, go_code

# Create user
add_api_code_docs(
    "POST",
    "/users",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

uid = client.add_user({"ANY": "DATA"})
"""
    ),
    js_code(
        """
import { MemoBaseClient } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);

const userId = await client.addUser({ANY: "DATA"});
"""
    ),
    go_code(
        """
import (
    "fmt"
    "log"

    "github.com/google/uuid"
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

    // Add a user
    userID := uuid.New().String()
    data := map[string]interface{}{"ANY": "DATA"}
    _, err = client.AddUser(data, userID)
    if err != nil {
        log.Fatalf("Failed to add user: %v", err)
    }
    fmt.Printf("User added with ID: %s\n", userID)
}
"""
    ),
)

# Get user
add_api_code_docs(
    "GET",
    "/users/{user_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

u = client.get_user(uid)
"""
    ),
    js_code(
        """
import { MemoBaseClient } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);

const user = await client.getUser(userId);
"""
    ),
    go_code(
        """
import (
    "fmt"
    "log"

    "github.com/google/uuid"
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
    fmt.Printf("Successfully retrieved user with ID: %s\n", user.UserID)
}
"""
    ),
)

# Update user
add_api_code_docs(
    "PUT",
    "/users/{user_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

client.update_user(uid, {"ANY": "NEW_DATA"})
"""
    ),
    js_code(
        """
import { MemoBaseClient } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);

await client.updateUser(userId, {ANY: "NEW_DATA"});
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

    // Update a user
    userID := "EXISTING_USER_ID" // Replace with an actual user ID
    newData := map[string]interface{}{"ANY": "NEW_DATA"}
    _, err = client.UpdateUser(userID, newData)
    if err != nil {
        log.Fatalf("Failed to update user: %v", err)
    }
    fmt.Printf("Successfully updated user with ID: %s\n", userID)
}
"""
    ),
)

# Delete user
add_api_code_docs(
    "DELETE",
    "/users/{user_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

client.delete_user(uid)
"""
    ),
    js_code(
        """
import { MemoBaseClient } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);

await client.deleteUser(userId);
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

    // Delete a user
    userID := "EXISTING_USER_ID" // Replace with an actual user ID
    err = client.DeleteUser(userID)
    if err != nil {
        log.Fatalf("Failed to delete user: %v", err)
    }
    fmt.Printf("Successfully deleted user with ID: %s\n", userID)
}
"""
    ),
)

# Get user context
add_api_code_docs(
    "GET",
    "/users/context/{user_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

context = u.context()
"""
    ),
    js_code(
        """
import { MemoBaseClient } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);
const user = await client.getUser(userId);

const context = await user.context();
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

    // Get user context
    userID := "EXISTING_USER_ID" // Replace with an actual user ID
    user, err := client.GetUser(userID, false)
    if err != nil {
        log.Fatalf("Failed to get user: %v", err)
    }

    context, err := user.Context(nil)
    if err != nil {
        log.Fatalf("Failed to get context: %v", err)
    }
    fmt.Printf("User context: %s\n", context)
}
"""
    ),
)
