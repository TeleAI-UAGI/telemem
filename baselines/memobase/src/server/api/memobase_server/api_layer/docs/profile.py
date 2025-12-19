from .basic_docs import add_api_code_docs, py_code, js_code, go_code

# Get user profile
add_api_code_docs(
    "GET",
    "/users/profile/{user_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

u = client.get_user(uid)
p = u.profile()
"""
    ),
    js_code(
        """
import { MemoBaseClient } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);
const user = await client.getUser(userId);

const profiles = await user.profile();
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

    // Get user profile
    profiles, err := user.Profile(nil)
    if err != nil {
        log.Fatalf("Failed to get user profile: %v", err)
    }

    // Print profiles
    fmt.Println("\nUser Profiles:")
    for _, profile := range profiles {
        fmt.Printf("ID: %s\nTopic: %s\nSub-topic: %s\nContent: %s\n\n",
            profile.ID,
            profile.Attributes.Topic,
            profile.Attributes.SubTopic,
            profile.Content,
        )
    }
}
"""
    ),
)

# Create user profile
add_api_code_docs(
    "POST",
    "/users/profile/{user_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

user = client.get_user('user_id')
user.add_profile(content="I am a software engineer", topic="career", sub_topic="job")
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

    // Add a profile
    profileID, err := user.AddProfile("value", "topic", "sub_topic")
    if err != nil {
        log.Fatalf("Failed to add profile: %v", err)
    }
    fmt.Printf("Successfully added profile with ID: %s\n", profileID)
}
"""
    ),
)

# Update user profile
add_api_code_docs(
    "PUT",
    "/users/profile/{user_id}/{profile_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

user = client.get_user('user_id')
user.update_profile(profile_id="profile_id", content="I am a software engineer", topic="career", sub_topic="job")
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

    // Update a profile
    profileID := "EXISTING_PROFILE_ID" // Replace with an actual profile ID
    err = user.UpdateProfile(profileID, "value2", "topic2", "sub_topic2")
    if err != nil {
        log.Fatalf("Failed to update profile: %v", err)
    }
    fmt.Printf("Successfully updated profile with ID: %s\n", profileID)
}
"""
    ),
)

# Delete user profile
add_api_code_docs(
    "DELETE",
    "/users/profile/{user_id}/{profile_id}",
    py_code(
        """
from memobase import MemoBaseClient

client = MemoBaseClient(project_url='PROJECT_URL', api_key='PROJECT_TOKEN')

user = client.get_user('user_id')
user.delete_profile('profile_id')
"""
    ),
    js_code(
        """
import { MemoBaseClient } from '@memobase/memobase';

const client = new MemoBaseClient(process.env.MEMOBASE_PROJECT_URL, process.env.MEMOBASE_API_KEY);

await client.deleteProfile('user_id', 'profile_id');
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

    // Delete a profile
    profileID := "EXISTING_PROFILE_ID" // Replace with an actual profile ID
    err = user.DeleteProfile(profileID)
    if err != nil {
        log.Fatalf("Failed to delete profile: %v", err)
    }
    fmt.Printf("Successfully deleted profile with ID: %s\n", profileID)
}
"""
    ),
)
