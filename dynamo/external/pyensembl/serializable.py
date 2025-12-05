# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class Serializable:
    """
    Base class for serializable objects.
    Requires subclasses to implement to_dict().
    Default from_dict() implementation passes dict as kwargs to constructor.
    Uses __reduce__ for pickling to ensure from_dict is used.
    """
    def to_dict(self):
        raise NotImplementedError("Subclasses must implement to_dict")

    @classmethod
    def from_dict(cls, state_dict):
        return cls(**state_dict)

    def __reduce__(self):
        return (self.__class__.from_dict, (self.to_dict(),))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        return not self.__eq__(other)

